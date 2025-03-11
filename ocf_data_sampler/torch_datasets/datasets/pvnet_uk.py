"""Torch dataset for UK PVNet."""

from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from typing_extensions import override

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS, RSS_MEAN, RSS_STD
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample import (
    convert_gsp_to_numpy_sample,
    convert_nwp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.select.geospatial import osgb_to_lon_lat
from ocf_data_sampler.torch_datasets.utils.merge_and_fill_utils import (
    fill_nans_in_arrays,
    merge_dicts,
)
from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
from ocf_data_sampler.torch_datasets.utils.validate_channels import (
    validate_nwp_channels,
    validate_satellite_channels,
)
from ocf_data_sampler.utils import minutes

xr.set_options(keep_attrs=True)


def process_and_combine_datasets(
    dataset_dict: dict,
    config: Configuration,
    t0: pd.Timestamp,
    location: Location,
) -> dict:
    """Normalise and convert data to numpy arrays."""
    numpy_modalities = []

    if "nwp" in dataset_dict:
        nwp_numpy_modalities = {}

        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            provider = config.input_data.nwp[nwp_key].provider

            # Standardise and convert to NumpyBatch
            da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

        # Combine the NWPs into NumpyBatch
        numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        da_sat = dataset_dict["sat"]

        # Standardise and convert to NumpyBatch
        da_sat = (da_sat - RSS_MEAN) / RSS_STD
        numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

    if "gsp" in dataset_dict:
        gsp_config = config.input_data.gsp
        da_gsp = dataset_dict["gsp"]
        da_gsp = da_gsp / da_gsp.effective_capacity_mwp

        # Convert to NumpyBatch
        numpy_modalities.append(
            convert_gsp_to_numpy_sample(
                da_gsp,
                t0_idx=-gsp_config.interval_start_minutes / gsp_config.time_resolution_minutes,
            ),
        )

    # Add GSP location data
    numpy_modalities.append(
        {
            GSPSampleKey.gsp_id: location.id,
            GSPSampleKey.x_osgb: location.x,
            GSPSampleKey.y_osgb: location.y,
        },
    )

    # Only add solar position if explicitly configured
    has_solar_config = (
        hasattr(config.input_data, "solar_position") and
        config.input_data.solar_position is not None
    )

    if has_solar_config:
        solar_config = config.input_data.solar_position

        # Create datetime range for solar position calculation
        datetimes = pd.date_range(
            t0 + minutes(solar_config.interval_start_minutes),
            t0 + minutes(solar_config.interval_end_minutes),
            freq=minutes(solar_config.time_resolution_minutes),
        )

        # Convert OSGB coordinates to lon/lat
        lon, lat = osgb_to_lon_lat(location.x, location.y)

        # Calculate solar positions and add to modalities
        solar_positions = make_sun_position_numpy_sample(datetimes, lon, lat)
        numpy_modalities.append(solar_positions)

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    combined_sample = fill_nans_in_arrays(combined_sample)

    return combined_sample


def compute(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays."""
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute(scheduler="single-threaded")
    return xarray_dict


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
        freq=minutes(config.input_data.gsp.time_resolution_minutes),
    )
    return valid_t0_times


def get_gsp_locations(gsp_ids: list[int] | None = None) -> list[Location]:
    """Get list of locations of all GSPs."""
    if gsp_ids is None:
        gsp_ids = list(range(1, 318))

    locations = []

    # Load UK GSP locations
    df_gsp_loc = pd.read_csv(
        files("ocf_data_sampler.data").joinpath("uk_gsp_locations.csv"),
        index_col="gsp_id",
    )

    for gsp_id in gsp_ids:
        locations.append(
            Location(
                coordinate_system="osgb",
                x=df_gsp_loc.loc[gsp_id].x_osgb,
                y=df_gsp_loc.loc[gsp_id].y_osgb,
                id=gsp_id,
            ),
        )
    return locations


class PVNetUKRegionalDataset(Dataset):
    """A torch Dataset for creating PVNet UK regional samples."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
        gsp_ids: list[int] | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet UK GSP samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
            gsp_ids: List of GSP IDs to create samples for. Defaults to all
        """
        # config = load_yaml_configuration(config_filename)
        config: Configuration = load_yaml_configuration(config_filename)
        validate_nwp_channels(config)
        validate_satellite_channels(config)

        datasets_dict = get_dataset_dict(config.input_data)

        # Get t0 times where all input data is available
        valid_t0_times = find_valid_t0_times(datasets_dict, config)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times <= pd.Timestamp(end_time)]

        # Construct list of locations to sample from
        locations = get_gsp_locations(gsp_ids)

        # Construct a lookup for locations - useful for users to construct sample by GSP ID
        location_lookup = {loc.id: loc for loc in locations}

        # Construct indices for sampling
        t_index, loc_index = np.meshgrid(
            np.arange(len(valid_t0_times)),
            np.arange(len(locations)),
        )

        # Make array of all possible (t0, location) coordinates. Each row is a single coordinate
        index_pairs = np.stack((t_index.ravel(), loc_index.ravel())).T

        # Assign coords and indices to self
        self.valid_t0_times = valid_t0_times
        self.locations = locations
        self.location_lookup = location_lookup
        self.index_pairs = index_pairs

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

    @override
    def __len__(self) -> int:
        return len(self.index_pairs)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = compute(sample_dict)

        sample = process_and_combine_datasets(sample_dict, self.config, t0, location)

        return sample

    @override
    def __getitem__(self, idx: int) -> dict:
        # Get the coordinates of the sample
        t_index, loc_index = self.index_pairs[idx]
        location = self.locations[loc_index]
        t0 = self.valid_t0_times[t_index]

        # Generate the sample
        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, gsp_id: int) -> dict:
        """Generate a sample for the given coordinates.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
            gsp_id: GSP ID
        """
        # Check the user has asked for a sample which we have the data for
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        if gsp_id not in self.location_lookup:
            raise ValueError(f"Input GSP '{gsp_id}' not known")

        location = self.location_lookup[gsp_id]

        return self._get_sample(t0, location)


class PVNetUKConcurrentDataset(Dataset):
    """A torch Dataset for creating concurrent PVNet UK regional samples."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
        gsp_ids: list[int] | None = None,
    ) -> None:
        """A torch Dataset for creating concurrent samples of PVNet UK regional data.

        Each concurrent sample includes the data from all GSPs for a single t0 time

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
            gsp_ids: List of all GSP IDs included in each sample. Defaults to all
        """
        config = load_yaml_configuration(config_filename)

        # Validate channels for NWP and satellite data
        validate_nwp_channels(config)
        validate_satellite_channels(config)

        datasets_dict = get_dataset_dict(config.input_data)

        # Get t0 times where all input data is available
        valid_t0_times = find_valid_t0_times(datasets_dict, config)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times <= pd.Timestamp(end_time)]

        # Construct list of locations to sample from
        locations = get_gsp_locations(gsp_ids)

        # Assign coords and indices to self
        self.valid_t0_times = valid_t0_times
        self.locations = locations

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp) -> dict:
        """Generate a concurrent PVNet sample for given init-time.

        Args:
            t0: init-time for sample
        """
        # Slice by time then load to avoid loading the data multiple times from disk
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = compute(sample_dict)

        gsp_samples = []

        # Prepare sample for each GSP
        for location in self.locations:
            gsp_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            gsp_numpy_sample = process_and_combine_datasets(
                gsp_sample_dict,
                self.config,
                t0,
                location,
            )
            gsp_samples.append(gsp_numpy_sample)

        # Stack GSP samples
        return stack_np_samples_into_batch(gsp_samples)

    @override
    def __getitem__(self, idx: int) -> dict:
        return self._get_sample(self.valid_t0_times[idx])

    def get_sample(self, t0: pd.Timestamp) -> dict:
        """Generate a sample for the given init-time.

        Useful for users to generate specific samples.

        Args:
            t0: init-time for sample
        """
        # Check data is availablle for init-time t0
        if t0 not in self.valid_t0_times:
            raise ValueError(f"Input init time '{t0!s}' not in valid times")
        return self._get_sample(t0)
