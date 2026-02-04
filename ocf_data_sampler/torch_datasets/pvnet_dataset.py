"""Torch dataset for PVNet."""

import logging
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from pydantic.warnings import UnsupportedFieldAttributeWarning
from torch.utils.data import Dataset
from typing_extensions import override

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample import (
    convert_xarray_dict_to_numpy_sample,
    encode_datetimes,
    get_t0_embedding,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.torch_datasets.utils import (
    add_alterate_coordinate_projections,
    config_normalization_values_to_dicts,
    diff_nwp_data,
    find_valid_time_periods,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.utils import load_data_dict, minutes

# Ignore pydantic warning which doesn't cause an issue
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

xr.set_options(keep_attrs=True)

logger = logging.getLogger(__name__)


class PickleCacheMixin:
    """A mixin for classes that need to cache their state using pickle."""

    def __init__(self, *args: list, **kwargs: dict) -> None:
        """Initialize the pickle path and call the parent constructor."""
        self._pickle_path = None
        super().__init__(*args, **kwargs)  # cooperative multiple inheritance

    def presave_pickle(self, pickle_path: str) -> None:
        """Save the full object state to a pickle file and store the pickle path."""
        self._pickle_path = pickle_path
        with open(pickle_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def __getstate__(self) -> dict:
        """If presaved, only pickle reference. Otherwise pickle everything."""
        if self._pickle_path:
            return {"_pickle_path": self._pickle_path}
        else:
            return self.__dict__

    def __setstate__(self, state: dict) -> None:
        """Restore object from pickle, reloading from presaved file if possible."""
        self.__dict__.update(state)
        if self._pickle_path and os.path.exists(self._pickle_path):
            with open(self._pickle_path, "rb") as f:
                saved_state = pickle.load(f)  # noqa: S301
                self.__dict__.update(saved_state)


def get_locations(generation_data: xr.DataArray) -> list[Location]:
    """Get list of locations of all locations.

    Args:
        generation_data: xarray dataarray of generation data with location info
    """
    locations = []
    location_ids = generation_data.location_id.values

    for location_id in location_ids:
        gen_data = generation_data.sel(location_id=location_id)
        locations.append(
            Location(
                x=gen_data.longitude.values,
                y=gen_data.latitude.values,
                coord_system="lon_lat",
                id=int(location_id),
            ),
        )

    return locations


class AbstractPVNetDataset(PickleCacheMixin, Dataset):
    """Abstract class for PVNet datasets."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A generic torch Dataset for creating PVNet samples.

        Contains methods to find valid times and locations
        and process and combine these sources for various data inputs
        common to both standard and concurrent PVNet datasets.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        super().__init__()

        config = load_yaml_configuration(config_filename)

        datasets_dict = get_dataset_dict(config.input_data)

        # Check if generation data has nans
        self.complete_generation = not datasets_dict["generation"].isnull().any()

        if self.complete_generation:
            valid_t0_times = self.find_valid_t0_times(datasets_dict, config)

            # Filter t0 times to given range
            if start_time is not None:
                valid_t0_times = valid_t0_times[valid_t0_times >= pd.Timestamp(start_time)]

            if end_time is not None:
                valid_t0_times = valid_t0_times[valid_t0_times <= pd.Timestamp(end_time)]

            self.valid_t0_times = valid_t0_times
        else:
            logger.info(
                "Generation data has nans so t0s are handled separately for each location_id.",
            )
            # If non-identical times per location, find valid t0s per location id
            valid_t0_and_location_ids = self.find_valid_t0_and_location_ids(datasets_dict, config)

            # Filter t0 times to given range
            if start_time is not None:
                valid_t0_and_location_ids = valid_t0_and_location_ids[
                    valid_t0_and_location_ids["t0"] >= pd.Timestamp(start_time)
                ]

            if end_time is not None:
                valid_t0_and_location_ids = valid_t0_and_location_ids[
                    valid_t0_and_location_ids["t0"] <= pd.Timestamp(end_time)
                ]
            self.valid_t0_and_location_ids = valid_t0_and_location_ids

        # Construct list of locations to sample from
        locations = get_locations(generation_data=datasets_dict["generation"])

        self.locations = add_alterate_coordinate_projections(locations, datasets_dict)

        # Assign config and input data to self
        self.config = config
        self.datasets_dict = datasets_dict

        # Assign t0 idx value
        self.t0_idx = (
            -config.input_data.generation.interval_start_minutes
            // config.input_data.generation.time_resolution_minutes
        )

        # Extract the normalisation values from the config for faster access
        means_dict, stds_dict = config_normalization_values_to_dicts(config)
        self.means_dict = means_dict
        self.stds_dict = stds_dict

    def process_and_combine_datasets(
        self,
        sample_dict: dict[str, xr.DataArray],
        t0: pd.Timestamp,
        location: Location,
    ) -> NumpySample:
        """Process and combine xarray datasets into a unified NumpySample.

        This is the unified conversion approach that replaces individual
        convert_*_to_numpy_sample functions.

        Args:
            sample_dict: Dictionary of xarray DataArrays by modality
            t0: The t0 timestamp for this sample
            location: The location for this sample

        Returns:
            NumpySample with all modalities combined
        """
        # Find t0 index in the generation data
        t0_idx = None
        if "generation" in sample_dict:
            gen_da = sample_dict["generation"]
            t0_idx = int(np.where(gen_da.time_utc.values == t0)[0][0])

        # Build dictionary of xarray data for unified conversion
        xarray_dict = {}

        # Add generation if present
        if "generation" in sample_dict:
            xarray_dict["generation"] = sample_dict["generation"]

        # Add NWP data (can be multiple sources)
        if "nwp" in sample_dict:
            xarray_dict["nwp"] = sample_dict["nwp"]

        # Add satellite if present (use 'satellite' key for unified converter)
        if "sat" in sample_dict:
            xarray_dict["satellite"] = sample_dict["sat"]

        # Use the unified converter - this replaces all the old convert_*_to_numpy_sample calls
        sample = convert_xarray_dict_to_numpy_sample(
            xarray_dict,
            t0_idx=t0_idx,
        )

        # Add location information
        sample["location_id"] = location.id
        if hasattr(location, "x"):
            sample["longitude"] = location.x
        if hasattr(location, "y"):
            sample["latitude"] = location.y

        # Add datetime encodings if time_utc is present (from generation data)
        # Must happen before removing datetime64 arrays
        if "time_utc" in sample:
            datetimes = pd.to_datetime(sample["time_utc"])
            datetime_sample = encode_datetimes(datetimes)
            sample.update(datetime_sample)

        # Add solar position if configured
        if self.config.input_data.solar_position is not None:
            solar_config = self.config.input_data.solar_position
            time_resolution = pd.Timedelta(solar_config.time_resolution_minutes, unit="minutes")
            start_offset = pd.Timedelta(solar_config.interval_start_minutes, unit="minutes")
            end_offset = pd.Timedelta(solar_config.interval_end_minutes, unit="minutes")
            datetimes = pd.date_range(
                start=t0 + start_offset,
                end=t0 + end_offset,
                freq=time_resolution,
            )

            solar_sample = make_sun_position_numpy_sample(
                datetimes=datetimes,
                lon=location.x,
                lat=location.y,
            )
            sample.update(solar_sample)

        # Add t0 embedding if configured
        if hasattr(self.config.input_data, "t0_embedding") and \
           self.config.input_data.t0_embedding is not None:
            t0_embedding_sample = get_t0_embedding(
                t0=t0,
                #periods=self.config.input_data.t0_embedding.periods,
                #(removed for test_collate.py for periods attribute)
                embeddings=self.config.input_data.t0_embedding.embeddings,
            )
            sample.update(t0_embedding_sample)

        # Clean up datetime64 arrays (must be last)
        for key, value in list(sample.items()):
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.datetime64):
                del sample[key]

        return sample

    @staticmethod
    def find_valid_t0_times(datasets_dict: dict, config: Configuration) -> pd.DatetimeIndex:
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
        datasets_dict: dict,
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
                pd.DatetimeIndex(generation.time_utc.values),
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
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        super().__init__(config_filename, start_time, end_time)

        # Construct a lookup for locations - useful for users to construct sample by location ID
        self.location_lookup = {loc.id: loc for loc in self.locations}

    @override
    def __len__(self) -> int:
        if self.complete_generation:
            return len(self.locations) * len(self.valid_t0_times)
        # For non-identical generation time periods all t0 and location combinations already present
        return len(self.valid_t0_and_location_ids)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> NumpySample:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        # Slice data by space and time
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = load_data_dict(sample_dict)
        sample_dict = diff_nwp_data(sample_dict, self.config)

        # Use unified conversion - this is the main change!
        # Previously this would call multiple convert_*_to_numpy_sample functions
        # Now it's all handled by process_and_combine_datasets
        sample = self.process_and_combine_datasets(sample_dict, t0, location)

        # Add t0 timestamp to sample (required by tests and batch processing)
        sample["t0"] = t0

        return sample

    @override
    def __getitem__(self, idx: int) -> NumpySample:
        # Get the coordinates of the sample
        if idx >= len(self):
            raise ValueError(f"Index {idx} out of range for dataset of length {len(self)}")

        if self.complete_generation:
            # t_index will be between 0 and len(self.valid_t0_times)-1
            t_index = idx % len(self.valid_t0_times)

            # For each location, there are len(self.valid_t0_times) possible samples
            loc_index = idx // len(self.valid_t0_times)

            location = self.locations[loc_index]
            t0 = self.valid_t0_times[t_index]
        else:
            # Get the coordinates of the sample
            t0, location_id = self.valid_t0_and_location_ids.iloc[idx]

            # Get location from location id
            location = self.location_lookup[location_id]

        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, location_id: int) -> NumpySample:
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

    def validate_sample_request(self, t0: pd.Timestamp, location_id: int) -> None:
        """Validate if a sample request for the given coordinates is valid."""
        if self.complete_generation:
            if t0 not in self.valid_t0_times:
                raise ValueError(f"Input init time '{t0!s}' not in valid times")
            if location_id not in self.location_lookup:
                raise ValueError(f"Input location '{location_id}' not known")
        else:
            if not (
                t0 in self.valid_t0_and_location_ids.index
                and self.valid_t0_and_location_ids.loc[t0, "location_id"] == location_id
            ):
                raise ValueError(
                    f"Input t0 time '{t0!s}' and location id '{location_id}' "
                    f"pair not in valid t0 and location pairs",
                )


class PVNetConcurrentDataset(AbstractPVNetDataset):
    """A torch Dataset for creating concurrent PVNet location samples."""

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp) -> NumpyBatch:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = load_data_dict(sample_dict)
        sample_dict = diff_nwp_data(sample_dict, self.config)

        samples = []

        # Prepare sample for each location using unified conversion
        for location in self.locations:
            sliced_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            numpy_sample = self.process_and_combine_datasets(
                sliced_sample_dict,
                t0,
                location,
            )
            samples.append(numpy_sample)

        # Stack samples into batch
        return stack_np_samples_into_batch(samples)

    @override
    def __getitem__(self, idx: int) -> NumpyBatch:
        return self._get_sample(self.valid_t0_times[idx])

    def get_sample(self, t0: pd.Timestamp) -> NumpyBatch:
        """Generate a sample for the given init-time.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
        """
        # Check data is available for init-time t0
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        return self._get_sample(t0)
