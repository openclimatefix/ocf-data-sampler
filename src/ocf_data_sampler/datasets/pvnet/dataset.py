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
from ocf_data_sampler.config.model import Configuration
from ocf_data_sampler.datasets.cache import PickleCacheMixin
from ocf_data_sampler.datasets.pvnet.loading import get_dataset_dict
from ocf_data_sampler.datasets.pvnet.materialise import load_data_dict
from ocf_data_sampler.datasets.pvnet.preprocess import (
    apply_dropout_to_datasets,
    config_normalization_values_to_dicts,
    diff_nwp_data,
    fill_nans_in_dataset_dicts,
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
from ocf_data_sampler.select import (
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.spatial import Location, convert_coordinates, find_coord_system

logger = logging.getLogger(__name__)



def get_locations(generation_data: xr.DataArray) -> list[Location]:
    """Get list of locations of all locations.

    Args:
        generation_data: xarray dataarray of generation data with location info
    """
    locations = []
    location_ids = generation_data["location_id"].values

    for location_id in location_ids:
        gen_data = generation_data.sel(location_id=location_id)
        locations.append(
            Location(
                x=gen_data["longitude"].values,
                y=gen_data["latitude"].values,
                coord_system="lon_lat",
                id=int(location_id),
            ),
        )

    return locations


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

    Args:
        times: DatetimeIndex of times to filter
        time_periods: List of tuples specifying the start and end times for each period
    """
    if len(time_periods)==0:
        raise ValueError("At least one time period must be provided")

    mask = np.full(len(times), False)

    for start_time, end_time in time_periods:

        start_time = times[0] if start_time is None else np.datetime64(start_time)
        end_time = times[-1] if end_time is None else np.datetime64(end_time)

        mask |= (times >= start_time) & (times <= end_time)

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

        datasets_dict = get_dataset_dict(config.input_data)

        # Check if generation data has nans
        self.complete_generation = not datasets_dict["generation"].isnull().any()

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
            valid_t0_and_location_ids = self.find_valid_t0_and_location_ids(datasets_dict, config)

            # Filter t0 times to given range
            if time_periods is not None:
                mask = get_time_periods_mask(valid_t0_and_location_ids["t0"], time_periods)
                valid_t0_and_location_ids = valid_t0_and_location_ids[mask]

            self.valid_t0_and_location_ids = valid_t0_and_location_ids

        # Construct list of locations to sample from
        locations = get_locations(generation_data=datasets_dict["generation"])

        self.locations = add_alternate_coordinate_projections(locations, datasets_dict)

        self.config = config
        self.include_extra_metadata = include_extra_metadata

        if use_xarray:
            self.datasets_dict = datasets_dict
        else:
            self.datasets_dict = xarray_to_lightarray_dict(datasets_dict)

        # Assign t0 idx value
        self.t0_idx = (
            -config.input_data.generation.interval_start_minutes
            // config.input_data.generation.time_resolution_minutes
        )

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

    def process_and_combine_datasets(
        self,
        dataset_dict: SourceDict,
        t0: np.datetime64,
        location: Location,
    ) -> NumpySample:
        """Normalise and convert data to numpy arrays.

        Args:
            dataset_dict: Dictionary of xarray datasets
            t0: init-time for sample
            location: location of the sample
        """
        # Normalise NWP
        if "nwp" in dataset_dict:
            for nwp_key, da_nwp in dataset_dict["nwp"].items():
                channel_means = self.mean_dict["nwp"][nwp_key]
                channel_stds = self.std_dict["nwp"][nwp_key]
                channel_mins = self.clip_min_dict["nwp"][nwp_key]
                channel_maxs = self.clip_max_dict["nwp"][nwp_key]
                dataset_dict["nwp"][nwp_key].data = (
                    (da_nwp.data.clip(channel_mins, channel_maxs) - channel_means)
                    / channel_stds
                )

        # Normalise satellite
        if "sat" in dataset_dict:
            channel_means = self.mean_dict["sat"]
            channel_stds = self.std_dict["sat"]
            channel_mins = self.clip_min_dict["sat"]
            channel_maxs = self.clip_max_dict["sat"]
            dataset_dict["sat"].data = (
                (dataset_dict["sat"].data.clip(channel_mins, channel_maxs) - channel_means)
                / channel_stds
            )

        # Fill NaNs
        dataset_dict = fill_nans_in_dataset_dicts(dataset_dict, config=self.config)

        # Convert all xarray modalities to a single NumpySample
        sample = convert_to_numpy_sample(dataset_dict, self.t0_idx, self.include_extra_metadata)

        # Add location metadata not present on the DataArray
        if "generation" in dataset_dict:
            sample["location_id"] = location.id

        # Add datetime encodings over the full generation time range
        generation_config = self.config.input_data.generation
        datetimes = date_range(
            t0 + minutes(generation_config.interval_start_minutes),
            t0 + minutes(generation_config.interval_end_minutes),
            freq=minutes(generation_config.time_resolution_minutes),
        )
        sample.update(encode_datetimes(datetimes=datetimes))

        # Add t0 embedding if configured
        if self.config.input_data.t0_embedding is not None:
            sample.update(
                make_t0_encoding_numpy_sample(t0, self.config.input_data.t0_embedding.embeddings),
            )

        # Add solar position if configured
        if self.config.input_data.solar_position is not None:
            solar_config = self.config.input_data.solar_position

            # Create datetime range for solar position calculation
            datetimes = date_range(
                t0 + minutes(solar_config.interval_start_minutes),
                t0 + minutes(solar_config.interval_end_minutes),
                freq=minutes(solar_config.time_resolution_minutes),
            )
            sample.update(
                make_sun_position_numpy_sample(
                    datetimes,
                    dataset_dict["generation"]["longitude"].values,
                    dataset_dict["generation"]["latitude"].values,
                ),
            )

        sample["t0"] = get_posix_timestamp(t0)

        return sample

    @staticmethod
    def find_valid_t0_times(
        datasets_dict: SourceDict,
        config: Configuration,
    ) -> NDArray[np.datetime64]:
        """Find the t0 times where all of the requested input data is available.

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        valid_time_periods = find_valid_time_periods(datasets_dict, config)

        # Fill out the contiguous time periods to get the t0 times
        valid_t0_times = fill_time_periods(
            valid_time_periods,
            freq=minutes(config.input_data.generation.time_resolution_minutes),
        )
        return valid_t0_times

    @staticmethod
    def find_valid_t0_and_location_ids(
        datasets_dict: SourceDict,
        config: Configuration,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available for each location.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        # Get valid time period for nwp and satellite
        datasets_without_generation = {k: v for k, v in datasets_dict.items() if k != "generation"}
        valid_time_periods = find_valid_time_periods(datasets_without_generation, config)

        # Loop over each location in system id and obtain valid periods
        generations = datasets_dict["generation"]
        location_ids = generations.location_id.values
        generation_config = config.input_data.generation
        valid_t0_and_location_ids = []
        for location_id in location_ids:
            generation = generations.sel(location_id=location_id)
            # Drop NaN values
            generation = generation.dropna(dim="time_utc")

            # Obtain valid time periods for this location
            time_periods = find_contiguous_t0_periods(
                generation["time_utc"].values,
                time_resolution=minutes(generation_config.time_resolution_minutes),
                interval_start=minutes(generation_config.interval_start_minutes),
                interval_end=minutes(generation_config.interval_end_minutes),
            )
            valid_time_periods_per_location = intersection_of_multiple_dataframes_of_periods(
                [valid_time_periods, time_periods],
            )

            # Fill out contiguous time periods to get t0 times
            valid_t0_times_per_location = fill_time_periods(
                valid_time_periods_per_location,
                freq=minutes(generation_config.time_resolution_minutes),
            )

            valid_t0_per_location = pd.DataFrame(index=valid_t0_times_per_location)
            valid_t0_per_location["location_id"] = location_id
            valid_t0_and_location_ids.append(valid_t0_per_location)

        valid_t0_and_location_ids = pd.concat(valid_t0_and_location_ids)
        return valid_t0_and_location_ids.reset_index(names="t0")


class PVNetDataset(AbstractPVNetDataset):
    """A torch Dataset for creating PVNet samples."""

    @override
    def __init__(
        self,
        config_filename: str,
        time_periods: list[tuple[None | str, None | str]] | None = None,
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
        # Apply dropout to the data sources in-place
        apply_dropout_to_datasets(sample_dict, t0, self.config)
        sample_dict = diff_nwp_data(sample_dict, self.config)
        return self.process_and_combine_datasets(sample_dict, t0, location)

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
        time_periods: list[tuple[None | str, None | str]] | None = None,
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
        # Apply dropout to the data sources in-place
        apply_dropout_to_datasets(sample_dict, t0, self.config)
        sample_dict = diff_nwp_data(sample_dict, self.config)

        samples = []

        # Prepare sample for each location
        for location in self.locations:
            sliced_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            numpy_sample = self.process_and_combine_datasets(
                sliced_sample_dict,
                t0,
                location,
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
