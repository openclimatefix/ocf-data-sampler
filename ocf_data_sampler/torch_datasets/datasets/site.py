"""Torch dataset for sites."""

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from typing_extensions import override

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.numpy_sample import (
    NWPSampleKey,
    convert_nwp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    convert_site_to_numpy_sample,
    encode_datetimes,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.torch_datasets.utils import (
    add_alterate_coordinate_projections,
    config_normalization_values_to_dicts,
    fill_nans_in_arrays,
    find_valid_time_periods,
    merge_dicts,
    slice_datasets_by_space,
    slice_datasets_by_time,
)
from ocf_data_sampler.utils import minutes, tensorstore_compute

xr.set_options(keep_attrs=True)


def get_locations(site_xr: xr.Dataset) -> list[Location]:
    """Get list of locations of all sites.

    Args:
        site_xr: xarray Dataset of site data
    """
    locations = []
    for site_id in site_xr.site_id.values:
        site = site_xr.sel(site_id=site_id)
        location = Location(
            id=site_id,
            x=site.longitude.values,
            y=site.latitude.values,
            coord_system="lon_lat",
        )
        locations.append(location)

    return locations

def process_and_combine_datasets(
    dataset_dict: dict,
    config: Configuration,
    t0: pd.Timestamp,
    means_dict: dict[str, xr.DataArray | dict[str, xr.DataArray]],
    stds_dict: dict[str, xr.DataArray | dict[str, xr.DataArray]],
) -> NumpySample:
    """Normalise and convert data to numpy arrays.

    Args:
        dataset_dict: Dictionary of xarray datasets
        config: Configuration object
        t0: init-time for sample
        means_dict: Nested dictionary of mean values for the input data sources
        stds_dict: Nested dictionary of std values for the input data sources
    """
    numpy_modalities = []

    if "nwp" in dataset_dict:
        nwp_numpy_modalities = {}

        for nwp_key, da_nwp in dataset_dict["nwp"].items():

            # Standardise and convert to NumpyBatch

            da_channel_means = means_dict["nwp"][nwp_key]
            da_channel_stds = stds_dict["nwp"][nwp_key]

            da_nwp = (da_nwp - da_channel_means) / da_channel_stds

            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

        # Combine the NWPs into NumpyBatch
        numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        da_sat = dataset_dict["sat"]

        # Standardise and convert to NumpyBatch
        da_channel_means = means_dict["sat"]
        da_channel_stds = stds_dict["sat"]

        da_sat = (da_sat - da_channel_means) / da_channel_stds

        numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))

    if "site" in dataset_dict:
        da_sites = dataset_dict["site"]
        da_sites = da_sites / da_sites.capacity_kwp

        # Convert to NumpyBatch
        numpy_modalities.append(convert_site_to_numpy_sample(da_sites))

        # add datetime features
        datetimes = pd.DatetimeIndex(da_sites.time_utc.values)
        datetime_features = encode_datetimes(datetimes=datetimes)

        numpy_modalities.append(datetime_features)

    # Only add solar position if explicitly configured
    if config.input_data.solar_position is not None:
        solar_config = config.input_data.solar_position

        # Create datetime range for solar position calculation
        datetimes = pd.date_range(
            t0 + minutes(solar_config.interval_start_minutes),
            t0 + minutes(solar_config.interval_end_minutes),
            freq=minutes(solar_config.time_resolution_minutes),
        )


        # Calculate solar positions and add to modalities
        numpy_modalities.append(
            make_sun_position_numpy_sample(
                datetimes,
                da_sites.longitude.values,
                da_sites.latitude.values,
                ),
            )

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    combined_sample = fill_nans_in_arrays(combined_sample)

    return combined_sample


class SitesDataset(Dataset):
    """A torch Dataset for creating PVNet Site samples."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet Site samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        config = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config.input_data)

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

        # Construct list of locations to sample from
        locations = get_locations(datasets_dict["site"])
        self.locations = add_alterate_coordinate_projections(
            locations,
            datasets_dict,
            primary_coords="lon_lat",
        )

        self.location_lookup = {loc.id: loc for loc in self.locations}

        # Get t0 times where all input data is available
        valid_t0_and_site_ids = self.find_valid_t0_and_site_ids(datasets_dict)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_and_site_ids = valid_t0_and_site_ids[
                valid_t0_and_site_ids["t0"] >= pd.Timestamp(start_time)
            ]

        if end_time is not None:
            valid_t0_and_site_ids = valid_t0_and_site_ids[
                valid_t0_and_site_ids["t0"] <= pd.Timestamp(end_time)
            ]

        # Assign coords and indices to self
        self.valid_t0_and_site_ids = valid_t0_and_site_ids

        # Extract the normalisation values from the config for faster access
        means_dict, stds_dict = config_normalization_values_to_dicts(config)
        self.means_dict = means_dict
        self.stds_dict = stds_dict

    def find_valid_t0_and_site_ids(
        self,
        datasets_dict: dict,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For each site location, find valid periods for that location

        Args:
            datasets_dict: A dictionary of input datasets
            config: Configuration file
        """
        # Get valid time period for nwp and satellite
        datasets_without_site = {k: v for k, v in datasets_dict.items() if k != "site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, self.config)

        # Loop over each location in system id and obtain valid periods
        sites = datasets_dict["site"]
        site_ids = sites.site_id.values
        site_config = self.config.input_data.site
        valid_t0_and_site_ids = []
        for site_id in site_ids:
            site = sites.sel(site_id=site_id)
            # Drop NaN values
            site = site.dropna(dim="time_utc")

            # Obtain valid time periods for this location
            time_periods = find_contiguous_t0_periods(
                pd.DatetimeIndex(site["time_utc"]),
                time_resolution=minutes(site_config.time_resolution_minutes),
                interval_start=minutes(site_config.interval_start_minutes),
                interval_end=minutes(site_config.interval_end_minutes),
            )
            valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
                [valid_time_periods, time_periods],
            )

            # Fill out contiguous time periods to get t0 times
            valid_t0_times_per_site = fill_time_periods(
                valid_time_periods_per_site,
                freq=minutes(site_config.time_resolution_minutes),
            )

            valid_t0_per_site = pd.DataFrame(index=valid_t0_times_per_site)
            valid_t0_per_site["site_id"] = site_id
            valid_t0_and_site_ids.append(valid_t0_per_site)

        valid_t0_and_site_ids = pd.concat(valid_t0_and_site_ids)
        valid_t0_and_site_ids.index.name = "t0"
        return valid_t0_and_site_ids.reset_index()

    @override
    def __len__(self) -> int:
        return len(self.valid_t0_and_site_ids)

    @override
    def __getitem__(self, idx: int) -> dict:
        # Get the coordinates of the sample
        t0, site_id = self.valid_t0_and_site_ids.iloc[idx]

        # get location from site id
        location = self.location_lookup[site_id]

        # Generate the sample
        return self._get_sample(t0, location)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates.

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)

        sample_dict = tensorstore_compute(sample_dict)

        return process_and_combine_datasets(
            sample_dict,
            self.config,
            t0,
            self.means_dict,
            self.stds_dict,
        )

    def get_sample(self, t0: pd.Timestamp, site_id: int) -> dict:
        """Generate a sample for a given site id and t0.

        Useful for users to generate samples by t0 and site id

        Args:
            t0: init-time for sample
            site_id: site id as int
        """
        location = self.location_lookup[site_id]

        return self._get_sample(t0, location)


class SitesDatasetConcurrent(Dataset):
    """A torch Dataset for creating PVNet Site batches with samples for all sites."""

    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> None:
        """A torch Dataset for creating PVNet Site samples.

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """
        config = load_yaml_configuration(config_filename)
        datasets_dict = get_dataset_dict(config.input_data)

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

        # get all locations
        self.locations = get_locations(datasets_dict["site"])

        # Get t0 times where all input data is available
        valid_t0s = self.find_valid_t0s(datasets_dict)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0s = valid_t0s[
                valid_t0s >= pd.Timestamp(start_time)
            ]

        if end_time is not None:
            valid_t0s = valid_t0s[
                valid_t0s <= pd.Timestamp(end_time)
            ]

        # Assign coords and indices to self
        self.valid_t0s = valid_t0s

        # Extract the normalisation values from the config for faster access
        means_dict, stds_dict = config_normalization_values_to_dicts(config)
        self.means_dict = means_dict
        self.stds_dict = stds_dict

    def find_valid_t0s(
        self,
        datasets_dict: dict,
    ) -> pd.DataFrame:
        """Find the t0 times where all of the requested input data is available.

        The idea is to
        1. Get valid time period for nwp and satellite
        2. For the first site location, find valid periods for that location
        Note there is an assumption that all sites have the same t0 data available

        Args:
            datasets_dict: A dictionary of input datasets
        """
        # Get valid time period for nwp and satellite
        datasets_without_site = {k: v for k, v in datasets_dict.items() if k != "site"}
        valid_time_periods = find_valid_time_periods(datasets_without_site, self.config)
        sites = datasets_dict["site"]

        # Taking just the first site value, assume t0s the same for all of them
        site_id = sites.site_id.values[0]
        site_config = self.config.input_data.site
        site = sites.sel(site_id=site_id)
        # Drop NaN values
        site = site.dropna(dim="time_utc")

        # Obtain valid time periods for this location
        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(site["time_utc"]),
            time_resolution=minutes(site_config.time_resolution_minutes),
            interval_start=minutes(site_config.interval_start_minutes),
            interval_end=minutes(site_config.interval_end_minutes),
        )
        valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
            [valid_time_periods, time_periods],
        )

        # Fill out contiguous time periods to get t0 times
        valid_t0_times_per_site = fill_time_periods(
            valid_time_periods_per_site,
            freq=minutes(site_config.time_resolution_minutes),
        )

        return valid_t0_times_per_site

    @override
    def __len__(self) -> int:
        return len(self.valid_t0s)

    @override
    def __getitem__(self, idx: int) -> dict:
        # Get the coordinates of the sample
        t0 = self.valid_t0s[idx]

        return self._get_batch(t0)

    def _get_batch(self, t0: pd.Timestamp) -> dict:
        """Generate the PVNet batch for given coordinates.

        Args:
            t0: init-time for sample
        """
        # slice by time first as we want to keep all site id info
        sample_dict = slice_datasets_by_time(self.datasets_dict, t0, self.config)
        sample_dict = tensorstore_compute(sample_dict)

        site_samples = []

        for location in self.locations:
            site_sample_dict = slice_datasets_by_space(sample_dict, location, self.config)
            site_numpy_sample = process_and_combine_datasets(
                site_sample_dict,
                self.config,
                t0,
                self.means_dict,
                self.stds_dict,
            )
            site_samples.append(site_numpy_sample)

        return stack_np_samples_into_batch(site_samples)


def coarsen_data(xr_data: xr.Dataset, coarsen_to_deg: float = 0.1) -> xr.Dataset:
    """Coarsen the data to a specified resolution in degrees.

    Args:
        xr_data: xarray dataset to coarsen
        coarsen_to_deg: resolution to coarsen to in degrees
    """
    if "latitude" in xr_data.coords and "longitude" in xr_data.coords:
        step = np.abs(xr_data.latitude.values[1] - xr_data.latitude.values[0])
        step = np.round(step, 4)
        coarsen_factor = int(coarsen_to_deg / step)
        if coarsen_factor > 1:
            xr_data = xr_data.coarsen(
                latitude=coarsen_factor,
                longitude=coarsen_factor,
                boundary="pad",
                coord_func="min",
            ).mean()

    return xr_data
