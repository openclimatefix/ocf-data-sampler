"""Torch dataset for sites"""
import logging

import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.select import (
    Location,
    fill_time_periods,
    find_contiguous_t0_periods,
    intersection_of_multiple_dataframes_of_periods,
    slice_datasets_by_time, slice_datasets_by_space
)
from ocf_data_sampler.utils import minutes
from ocf_data_sampler.torch_datasets.process_and_combine import process_and_combine_site_sample_dict
from ocf_data_sampler.torch_datasets.valid_time_periods import find_valid_time_periods

xr.set_options(keep_attrs=True)


def find_valid_t0_and_site_ids(
    datasets_dict: dict,
    config: Configuration,
) -> pd.DataFrame:
    """Find the t0 times where all of the requested input data is available

    The idea is to
    1. Get valid time period for nwp and satellite
    2. For each site location, find valid periods for that location

    Args:
        datasets_dict: A dictionary of input datasets
        config: Configuration file
    """

    # 1. Get valid time period for nwp and satellite
    datasets_nwp_and_sat_dict = {"nwp": datasets_dict["nwp"], "sat": datasets_dict["sat"]}
    valid_time_periods = find_valid_time_periods(datasets_nwp_and_sat_dict, config)

    # 2. Now lets loop over each location in system id and find the valid periods
    # Should we have a different option if there are not nans
    sites = datasets_dict["site"]
    site_ids = sites.site_id.values
    site_config = config.input_data.site
    valid_t0_and_site_ids = []
    for site_id in site_ids:
        site = sites.sel(site_id=site_id)

        # drop any nan values
        # not sure this is right?
        site = site.dropna(dim='time_utc')

        # Get the valid time periods for this location
        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(site["time_utc"]),
            sample_period_duration=minutes(site_config.time_resolution_minutes),
            interval_start=minutes(site_config.interval_start_minutes),
            interval_end=minutes(site_config.interval_end_minutes),
        )
        valid_time_periods_per_site = intersection_of_multiple_dataframes_of_periods(
            [valid_time_periods, time_periods]
        )

        # Fill out the contiguous time periods to get the t0 times
        valid_t0_times_per_site = fill_time_periods(
            valid_time_periods_per_site,
            freq=minutes(site_config.time_resolution_minutes)
        )

        valid_t0_per_site = pd.DataFrame(index=valid_t0_times_per_site)
        valid_t0_per_site['site_id'] = site_id
        valid_t0_and_site_ids.append(valid_t0_per_site)

    valid_t0_and_site_ids = pd.concat(valid_t0_and_site_ids)
    valid_t0_and_site_ids.index.name = 't0'
    valid_t0_and_site_ids.reset_index(inplace=True)

    return valid_t0_and_site_ids


def get_locations(site_xr: xr.Dataset):
    """Get list of locations of all sites"""

    locations = []
    for site_id in site_xr.site_id.values:
        site = site_xr.sel(site_id=site_id)
        location = Location(
            id=site_id,
            x=site.longitude.values,
            y=site.latitude.values,
            coordinate_system="lon_lat"
        )
        locations.append(location)

    return locations


class SitesDataset(Dataset):
    def __init__(
        self,
        config_filename: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ):
        """A torch Dataset for creating PVNet Site samples

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
        """

        config = load_yaml_configuration(config_filename)

        datasets_dict = get_dataset_dict(config)

        # get all locations
        self.locations = get_locations(datasets_dict['site'])

        # Get t0 times where all input data is available
        valid_t0_and_site_ids = find_valid_t0_and_site_ids(datasets_dict, config)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_and_site_ids \
                = valid_t0_and_site_ids[valid_t0_and_site_ids['t0'] >= pd.Timestamp(start_time)]

        if end_time is not None:
            valid_t0_and_site_ids \
                = valid_t0_and_site_ids[valid_t0_and_site_ids['t0'] <= pd.Timestamp(end_time)]


        # Assign coords and indices to self
        self.valid_t0_and_site_ids = valid_t0_and_site_ids

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config

    def __len__(self):
        return len(self.valid_t0_and_site_ids)

    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates

        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)

        sample = process_and_combine_site_sample_dict(sample_dict, self.config)
        sample = sample.compute()
        return sample

    def get_location_from_site_id(self, site_id):
        """Get location from system id"""

        locations = [loc for loc in self.locations if loc.id == site_id]
        if len(locations) == 0:
            raise ValueError(f"Location not found for site_id {site_id}")

        if len(locations) > 1:
            logging.warning(f"Multiple locations found for site_id {site_id}, but will take the first")

        return locations[0]

    def __getitem__(self, idx):

        # Get the coordinates of the sample
        t0, site_id = self.valid_t0_and_site_ids.iloc[idx]

        # get location from site id
        location = self.get_location_from_site_id(site_id)

        # Generate the sample
        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, site_id: int) -> dict:
        """Generate a sample for a given site id and t0.

        Useful for users to generate samples by t0 and site id

        Args:
            t0: init-time for sample
            site_id: site id as int
        """

        location = self.get_location_from_site_id(site_id)

        return self._get_sample(t0, location)
