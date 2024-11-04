"""Torch dataset for sites"""
import logging

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from ocf_data_sampler.select.find_contiguous_time_periods import (
    find_contiguous_t0_periods, find_contiguous_t0_periods_nwp,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.select.fill_time_periods import fill_time_periods

from ocf_data_sampler.config import Configuration, load_yaml_configuration

from ocf_data_sampler.select.location import Location

from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.select.time_slice_for_dataset import slice_datasets_by_time
from ocf_data_sampler.select.space_slice_for_dataset import slice_datasets_by_space
from ocf_data_sampler.torch_datasets.xarray_compute import compute
from ocf_data_sampler.torch_datasets.process_and_combine import process_and_combine_datasets
from ocf_data_sampler.time_functions import minutes


xr.set_options(keep_attrs=True)


def find_valid_t0_and_site_ids(
        datasets_dict: dict,
        config: Configuration,
) -> pd.DataFrame:
    """Find the t0 times where all of the requested input data is available

    The idea is to
    1. Get valid time periods for nwp
    2. Get valid time periods for satellite
    3. Get valid time period for nwp and satellite
    4. For each site location, find valid periods for that location

    Args:
        datasets_dict: A dictionary of input datasets
        config: Configuration file
    """

    assert set(datasets_dict.keys()).issubset({"nwp", "sat", "site"})

    contiguous_time_periods: dict[str: pd.DataFrame] = {}  # Used to store contiguous time periods from each data source

    # TODO refactor as this code is duplicated
    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.input_data.nwp.items():

            da = datasets_dict["nwp"][nwp_key]

            if nwp_config.dropout_timedeltas_minutes is None:
                max_dropout = minutes(0)
            else:
                max_dropout = minutes(np.max(np.abs(nwp_config.dropout_timedeltas_minutes)))

            if nwp_config.max_staleness_minutes is None:
                max_staleness = None
            else:
                max_staleness = minutes(nwp_config.max_staleness_minutes)

            # The last step of the forecast is lost if we have to diff channels
            if len(nwp_config.nwp_accum_channels) > 0:
                end_buffer = pd.to_timedelta(nwp_config.time_resolution_minutes)
            else:
                end_buffer =minutes(0)

            # This is the max staleness we can use considering the max step of the input data
            max_possible_staleness = (
                    pd.Timedelta(da["step"].max().item())
                    - minutes(nwp_config.forecast_minutes)
                    - end_buffer
            )

            # Default to use max possible staleness unless specified in config
            if max_staleness is None:
                max_staleness = max_possible_staleness
            else:
                # Make sure the max acceptable staleness isn't longer than the max possible
                assert max_staleness <= max_possible_staleness

            time_periods = find_contiguous_t0_periods_nwp(
                datetimes=pd.DatetimeIndex(da["init_time_utc"]),
                history_duration=minutes(nwp_config.history_minutes),
                max_staleness=max_staleness,
                max_dropout=max_dropout,
            )

            contiguous_time_periods[f'nwp_{nwp_key}'] = time_periods

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(datasets_dict["sat"]["time_utc"]),
            sample_period_duration=minutes(sat_config.time_resolution_minutes),
            history_duration=minutes(sat_config.history_minutes),
            forecast_duration=minutes(sat_config.forecast_minutes),
        )

        contiguous_time_periods['sat'] = time_periods

    # just get the values (not the keys)
    contiguous_time_periods_values = list(contiguous_time_periods.values())

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_periods_values) > 1:
        valid_time_periods = intersection_of_multiple_dataframes_of_periods(
            contiguous_time_periods_values
        )
    else:
        valid_time_periods = contiguous_time_periods_values[0]

    # check there are some valid time periods
    if len(valid_time_periods) == 0:
        raise ValueError(f"No valid time periods found, {contiguous_time_periods=}")

    # 4. Now lets loop over each location in system id and find the valid periods
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
            history_duration=minutes(site_config.history_minutes),
            forecast_duration=minutes(site_config.forecast_minutes),
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


def get_locations(site_xr:xr.Dataset):
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
            gsp_ids: list[int] | None = None,
    ):
        """A torch Dataset for creating PVNet UK GSP samples

        Args:
            config_filename: Path to the configuration file
            start_time: Limit the init-times to be after this
            end_time: Limit the init-times to be before this
            gsp_ids: List of GSP IDs to create samples for. Defaults to all
        """

        config = load_yaml_configuration(config_filename)

        datasets_dict = get_dataset_dict(config)

        # get all locations
        self.locations = get_locations(datasets_dict['site'])

        # Get t0 times where all input data is available
        valid_t0_and_site_ids = find_valid_t0_and_site_ids(datasets_dict, config)

        # Filter t0 times to given range

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
        sample_dict = compute(sample_dict)

        sample = process_and_combine_datasets(sample_dict, self.config, t0, location, sun_position_key='site')

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
        # TOD change to system ids
        t0, site_id = self.valid_t0_and_site_ids.iloc[idx]

        # get location from site id
        location = self.get_location_from_site_id(site_id)

        # Generate the sample
        return self._get_sample(t0, location)

    def get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate a sample for the given coordinates.

        Useful for users to generate samples by t0 and location

        Args:
            t0: init-time for sample
            location: location object
        """
        # Check the user has asked for a sample which we have the data for
        # TODO

        return self._get_sample(t0, location)