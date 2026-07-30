"""Torch dataset for PVNet."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from torch.utils.data import Dataset, default_collate
from typing_extensions import override

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.time_utils import date_range, get_posix_timestamp, minutes
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.cache import PickleCacheMixin
from ocf_data_sampler.datasets.pvnet.loading import get_dataset_dict
from ocf_data_sampler.datasets.pvnet.materialise import load_data_dict
from ocf_data_sampler.datasets.pvnet.preprocess import (
    config_normalization_values_to_dicts,
    preprocess_dataset_dict,
)
from ocf_data_sampler.datasets.pvnet.sample import (
    convert_to_numpy_sample,
    make_sun_position_numpy_sample,
    make_t0_encoding_numpy_sample,
)
from ocf_data_sampler.datasets.pvnet.slicing import (
    reduce_spatial_extent_of_datasets,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.datasets.pvnet.types import NumpySample, SourceDict, TensorBatch
from ocf_data_sampler.datasets.pvnet.valid_t0s import find_valid_time_periods
from ocf_data_sampler.features.time_encodings import encode_datetimes
from ocf_data_sampler.load import open_locations
from ocf_data_sampler.select import (
    fill_time_periods,
    find_contiguous_t0_periods,
    intersect_time_periods,
)
from ocf_data_sampler.spatial import Location, convert_coordinates, find_coord_system

logger = logging.getLogger(__name__)



def get_locations(csv_path: str, exclude_ids: list[int] | None = None) -> list[Location]:
    """Load the locations metadata and build the list of all locations.

    Args:
        csv_path: Path to the locations CSV data
        exclude_ids: Location IDs to drop from the returned locations
    """
    locations_data = open_locations(csv_path)

    if exclude_ids:
        missing_ids = np.setdiff1d(exclude_ids, locations_data["location_id"].values)
        if len(missing_ids) > 0:
            raise ValueError(
                f"Cannot exclude location IDs which are not in the locations data: {missing_ids}",
            )

        locations_data = locations_data[~locations_data["location_id"].isin(exclude_ids)]

        if len(locations_data) == 0:
            raise ValueError("All location IDs in the locations data have been excluded")

    return [
        Location(
            x=row.longitude,
            y=row.latitude,
            coord_system="lon_lat",
            id=int(row.location_id),
        )
        for row in locations_data.itertuples()
    ]


def xarray_to_lightarray_dict(
    dataset_dict: SourceDict[xr.DataArray],
) -> SourceDict[LightDataArray]:
    """Create a dictionary LightDataArrays from a dictionary of xarray datasets."""
    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_dataset_dict[k] = xarray_to_lightarray_dict(v)
        elif isinstance(v, xr.DataArray):
            new_dataset_dict[k] = LightDataArray.from_xarray(v)
        else:
            raise ValueError(f"Unexpected type ({type(v)})")
    return new_dataset_dict


def get_time_periods_mask(
    times: NDArray[np.datetime64],
    time_periods: list[tuple[str | None, str | None]],
) -> np.ndarray:
    """Get a boolean mask showing which times fall within any of the specified time periods.

    A `None` bound means the period is unbounded in that direction.

    Args:
        times: Array of times to filter
        time_periods: List of tuples specifying the start and end times for each period
    """
    if len(time_periods)==0:
        raise ValueError("At least one time period must be provided")

    mask = np.full(len(times), False)

    for start_time, end_time in time_periods:

        this_period_mask = np.full(len(times), True)

        # Inclusive of start_time, exclusive of end_time
        if start_time is not None:
            this_period_mask &= times >= np.datetime64(start_time)
        if end_time is not None:
            this_period_mask &= times < np.datetime64(end_time)

        mask |= this_period_mask

    return mask


def add_alternate_coordinate_projections(
    locations: list[Location],
    datasets_dict: SourceDict,
) -> list[Location]:
    """Add (in-place) coordinate projections for all dataset to a set of locations.

    Args:
        locations: A list of locations
        datasets_dict: The dataset dict to add projections for

    Returns:
        List of locations with all coordinate projections added
    """
    xs, ys = np.array([loc.in_coord_system("lon_lat") for loc in locations]).T

    datasets_list = []
    if "nwp" in datasets_dict:
        datasets_list.extend(datasets_dict["nwp"].values())
    if "sat" in datasets_dict:
        datasets_list.append(datasets_dict["sat"])

    computed_coord_systems = {"lon_lat"}

    # Find all the coord systems required by all datasets
    for da in datasets_list:

        # Find the coordinate system required by this dataset
        coord_system, *_ = find_coord_system(da)

        # Skip if the projections in this coord system have already been computed
        if coord_system not in computed_coord_systems:

            # If using geostationary coords we need to extract the area spec
            area_spec = da.attrs["area"] if coord_system=="geostationary" else None

            new_xs, new_ys = convert_coordinates(
                x=xs,
                y=ys,
                from_coords="lon_lat",
                target_coords=coord_system,
                area_spec=area_spec,
            )

            # Add the projection to the locations objects
            for x, y, loc in zip(new_xs, new_ys, locations, strict=True):
                loc.add_coord_system(x, y, coord_system)

            computed_coord_systems.add(coord_system)

    return locations


def build_numpy_sample(
    dataset_dict: SourceDict,
    t0: np.datetime64,
    location: Location,
    config: PVNetDataConfig,
    include_extra_metadata: bool = False,
) -> NumpySample:
    """Convert data to numpy arrays and add auxiliary features.

    Note: the data in `dataset_dict` is expected to already be preprocessed - see
    `preprocess_dataset_dict`.

    Args:
        dataset_dict: Dictionary of xarray datasets
        t0: init-time for sample
        location: location of the sample
        config: PVNetDataConfig object
        include_extra_metadata: Whether to add additional non-essential metadata to the sample
    """
    # Convert all xarray modalities to a single NumpySample
    sample = convert_to_numpy_sample(dataset_dict, include_extra_metadata)

    sample["location_id"] = location.id
    lon, lat = location.in_coord_system("lon_lat")

    if include_extra_metadata:
        sample["location_longitude"] = lon
        sample["location_latitude"] = lat

    # Add t0 embedding if configured
    if config.t0_embedding is not None:
        sample.update(
            make_t0_encoding_numpy_sample(t0, config.t0_embedding.embeddings),
        )

    # Add datetime encodings if configured
    if config.datetime_encoding is not None:
        dt_config = config.datetime_encoding

        datetimes = date_range(
            t0 + minutes(dt_config.interval_start_minutes),
            t0 + minutes(dt_config.interval_end_minutes),
            freq=minutes(dt_config.time_resolution_minutes),
        )
        sample.update(encode_datetimes(datetimes=datetimes))

    # Add solar position if configured
    if config.solar_position is not None:
        solar_config = config.solar_position

        # Create datetime range for solar position calculation
        datetimes = date_range(
            t0 + minutes(solar_config.interval_start_minutes),
            t0 + minutes(solar_config.interval_end_minutes),
            freq=minutes(solar_config.time_resolution_minutes),
        )

        sample.update(make_sun_position_numpy_sample(datetimes, lon=lon, lat=lat))

    sample["t0"] = get_posix_timestamp(t0)

    return sample


class AbstractPVNetDataset(PickleCacheMixin, Dataset):
    """Abstract class for PVNet datasets."""

    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        """A generic torch Dataset for creating PVNet samples.

        Contains methods to find valid times and locations
        and process and combine these sources for various data inputs
        common to both standard and concurrent PVNet datasets.

        Args:
            config_filename: Path to the configuration file
            time_periods: List of tuples specifying the start and end times for each period
            include_extra_metadata: Whether to include non-essential metadata for each sample in the
                sample dict.
            use_xarray: Whether to use xarray.DataArray or LightDataArray as the underlying data
                structure when sampling

        """
        super().__init__()

        config = load_yaml_configuration(config_filename)

        locations = get_locations(
            config.sampling_grid.locations_csv_path,
            config.sampling_grid.exclude_location_ids,
        )

        datasets_dict = get_dataset_dict(config)

        if "generation" in datasets_dict:
            location_ids = [loc.id for loc in locations]
            missing = np.setdiff1d(location_ids, datasets_dict["generation"]["location_id"].values)
            if len(missing) > 0:
                raise ValueError(f"Generation data is missing for location IDs: {missing}")

            # Slice the generation data to only include the specified locations. This allows us to
            # quality check the generation data for nans and find valid t0 times for each location.
            datasets_dict["generation"] = datasets_dict["generation"].sel(location_id=location_ids)

        # Check if generation data has nans. If generation isn't configured at all, there's no
        # per-location data availability to consider, so a single global t0 grid still applies.
        self.complete_generation = (
            "generation" not in datasets_dict or not datasets_dict["generation"].isnull().any()
        )

        if self.complete_generation:
            valid_t0_times = self.find_valid_t0_times(datasets_dict, config)

            # Filter t0 times to given range
            if time_periods is not None:
                mask = get_time_periods_mask(valid_t0_times, time_periods)
                valid_t0_times = valid_t0_times[mask]

            self.valid_t0_times = valid_t0_times
        else:
            logger.info(
                "Generation data has nans so t0s are handled separately for each location_id.",
            )
            # If non-identical times per location, find valid t0s per location id
            valid_t0_and_location_ids = self.find_valid_t0_and_location_ids(
                datasets_dict, locations, config,
            )

            # Filter t0 times to given range
            if time_periods is not None:
                mask = get_time_periods_mask(valid_t0_and_location_ids["t0"].values, time_periods)
                valid_t0_and_location_ids = valid_t0_and_location_ids.iloc[mask]

            self.valid_t0_and_location_ids = valid_t0_and_location_ids

        self.locations = add_alternate_coordinate_projections(locations, datasets_dict)

        self.config = config
        self.include_extra_metadata = include_extra_metadata

        if use_xarray:
            self.datasets_dict = datasets_dict
        else:
            self.datasets_dict = xarray_to_lightarray_dict(datasets_dict)

        # Extract the normalisation values from the config for faster access
        mean_dict, std_dict, clip_min_dict, clip_max_dict = (
            config_normalization_values_to_dicts(config)
        )
        self.mean_dict = mean_dict
        self.std_dict = std_dict
        self.clip_min_dict = clip_min_dict
        self.clip_max_dict = clip_max_dict

    def _sanitise_index(self, idx: int) -> int:
        """Sanitise dataset indexing and raise IndexError for out-of-range indices."""
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise TypeError(f"Dataset indices must be integers, got {type(idx)!r}")

        index = int(idx)
        n_samples = len(self)
        if index < 0:
            index += n_samples

        if index < 0 or index >= n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of length {n_samples}")

        return index

    @staticmethod
    def find_valid_t0_times(
        datasets_dict: SourceDict,
        config: PVNetDataConfig,
    ) -> NDArray[np.datetime64]:
        """Find the t0 times where all of the requested input data is available.

        Args:
            datasets_dict: A dictionary of input datasets
            config: PVNetDataConfig file
        """
        valid_time_periods = find_valid_time_periods(datasets_dict, config)

        # Fill out the contiguous time periods to get the t0 times
        valid_t0_times = fill_time_periods(
            valid_time_periods,
            freq=minutes(config.sampling_grid.t0_resolution_minutes),
        )
        return valid_t0_times

    @staticmethod
    def find_valid_t0_and_location_ids(
        datasets_dict: SourceDict,
        locations: list[Location],
        config: PVNetDataConfig,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available for each location.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            locations: The locations to find valid t0 times for
            config: PVNetDataConfig file
        """
        # Get valid time period for inputs other than generation
        non_gen_time_periods = find_valid_time_periods(
            datasets_dict={k: v for k, v in datasets_dict.items() if k != "generation"},
            config=config,
        )

        # There are separate input and target generation slices
        generation_windows = [
            w for w in (config.generation.input, config.generation.target) if w is not None
        ]

        # Loop over each location in system id and obtain valid periods
        valid_t0_and_location_ids: list[pd.DataFrame] = []
        for location in locations:
            # Drop NaN values for location
            generation = (
                datasets_dict["generation"]
                .sel(location_id=location.id)
                .dropna(dim="time_utc")
            )

            # Obtain valid time periods for this location for both input and target generation
            gen_time_periods = [
                find_contiguous_t0_periods(
                    generation["time_utc"].values,
                    time_resolution=minutes(config.generation.time_resolution_minutes),
                    interval_start=minutes(window_config.interval_start_minutes),
                    interval_end=minutes(window_config.interval_end_minutes),
                )
                for window_config in generation_windows
            ]
            valid_time_periods = intersect_time_periods(
                [non_gen_time_periods, *gen_time_periods],
            )

            # Fill out contiguous time periods to get t0 times
            valid_t0_times = fill_time_periods(
                valid_time_periods,
                freq=minutes(config.sampling_grid.t0_resolution_minutes),
            )

            valid_t0_and_location_ids.append(
                pd.DataFrame({"t0": valid_t0_times, "location_id": location.id})
            )

        return pd.concat(valid_t0_and_location_ids, ignore_index=True)


class PVNetDataset(AbstractPVNetDataset):
    """A torch Dataset for creating PVNet samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        super().__init__(config_filename, time_periods, include_extra_metadata, use_xarray)
        # Construct a lookup for locations - useful for users to construct sample by location ID
        self.location_lookup = {loc.id: loc for loc in self.locations}

    @override
    def __len__(self) -> int:
        if self.complete_generation:
            return len(self.locations) * len(self.valid_t0_times)
        # For non-identical generation time periods all t0 and location combinations already present
        return len(self.valid_t0_and_location_ids)

    def _get_sample(self, t0: np.datetime64, location: Location) -> NumpySample:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = load_data_dict(sample_dict)
        sample_dict = preprocess_dataset_dict(
            sample_dict, t0, self.config,
            self.mean_dict, self.std_dict, self.clip_min_dict, self.clip_max_dict,
        )
        return build_numpy_sample(
            sample_dict, t0, location, self.config, self.include_extra_metadata,
        )

    @override
    def __getitem__(self, idx: int) -> NumpySample:
        idx = self._sanitise_index(idx)

        # Get the coordinates of the sample
        if self.complete_generation:
            # t_index will be between 0 and len(self.valid_t0_times)-1
            t_index = idx % len(self.valid_t0_times)

            # For each location, there are len(self.valid_t0_times) possible samples
            loc_index = idx // len(self.valid_t0_times)

            location = self.locations[loc_index]
            t0 = self.valid_t0_times[t_index]
        else:
            # Get the coordinates of the sample
            t0 = self.valid_t0_and_location_ids["t0"].values[idx]
            location_id = self.valid_t0_and_location_ids["location_id"].values[idx]

            # Get location from location id
            location = self.location_lookup[location_id]

        return self._get_sample(t0, location)

    def get_sample(self, t0: np.datetime64, location_id: int) -> NumpySample:
        """Generate a sample for the given coordinates.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
            location_id: id for location
        """
        # Check the user has asked for a sample which we have the data for
        self.validate_sample_request(t0, location_id)

        location = self.location_lookup[location_id]

        return self._get_sample(t0, location)

    def validate_sample_request(self, t0: np.datetime64, location_id: int) -> None:
        """Validate if a sample request for the given coordinates is valid."""
        if self.complete_generation:
            if t0 not in self.valid_t0_times:
                raise ValueError(f"Input init time '{t0!s}' not in valid times")
            if location_id not in self.location_lookup:
                raise ValueError(f"Input location '{location_id}' not known")
        else:
            t0_idxs = self.valid_t0_and_location_ids["t0"]==t0
            if location_id not in self.valid_t0_and_location_ids[t0_idxs]["location_id"].values:
                raise ValueError(
                    f"Input t0 time '{t0!s}' and location id '{location_id}' "
                    f"pair not in valid t0 and location pairs",
                )


class PVNetConcurrentDataset(AbstractPVNetDataset):
    """A torch Dataset for creating concurrent PVNet location samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[str | None, str | None]] | None = None,
        include_extra_metadata: bool = False,
        use_xarray: bool = True,
    ) -> None:
        super().__init__(config_filename, time_periods, include_extra_metadata, use_xarray)

        if not self.complete_generation:
            raise NotImplementedError(
                "Concurrent PVNet dataset cannot be created when generation data is incomplete.",
            )

        self.datasets_dict = reduce_spatial_extent_of_datasets(
            self.datasets_dict,
            self.locations,
            self.config,
        )

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_times)

    def _get_sample(self, t0: np.datetime64) -> TensorBatch:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = load_data_dict(sample_dict)
        # Preprocessing is location-independent, so do it once before slicing per-location below
        sample_dict = preprocess_dataset_dict(
            sample_dict, t0, self.config,
            self.mean_dict, self.std_dict, self.clip_min_dict, self.clip_max_dict,
        )

        samples = []

        # Prepare sample for each location
        for location in self.locations:
            sliced_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            numpy_sample = build_numpy_sample(
                sliced_sample_dict, t0, location, self.config, self.include_extra_metadata,
            )
            samples.append(numpy_sample)

        # Stack samples
        return default_collate(samples)

    @override
    def __getitem__(self, idx: int) -> TensorBatch:
        idx = self._sanitise_index(idx)
        return self._get_sample(self.valid_t0_times[idx])

    def get_sample(self, t0: np.datetime64) -> TensorBatch:
        """Generate a sample for the given init-time.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
        """
        # Check data is available for init-time t0
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        return self._get_sample(t0)
