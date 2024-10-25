"""Torch dataset for PVNet"""

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import pkg_resources

from ocf_data_sampler.select.find_contiguous_time_periods import (
    find_contiguous_t0_periods, find_contiguous_t0_periods_nwp, 
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.select.fill_time_periods import fill_time_periods

from ocf_data_sampler.config import Configuration, load_yaml_configuration

from ocf_data_sampler.select.location import Location

from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.torch_datasets.process_and_combine import process_and_combine_datasets
from ocf_data_sampler.select.space_slice_for_dataset import slice_datasets_by_space
from ocf_data_sampler.select.time_slice_for_dataset import slice_datasets_by_time
from ocf_data_sampler.torch_datasets.xarray_compute import compute

xr.set_options(keep_attrs=True)

def minutes(minutes: list[float]):
    """Timedelta minutes

    Args:
        m: minutes
    """
    return pd.to_timedelta(minutes, unit="m")


def find_valid_t0_times(
    datasets_dict: dict,
    config: Configuration,
):
    """Find the t0 times where all of the requested input data is available

    Args:
        datasets_dict: A dictionary of input datasets 
        config: Configuration file
    """

    assert set(datasets_dict.keys()).issubset({"nwp", "sat", "gsp"})

    contiguous_time_periods: dict[str: pd.DataFrame] = {}  # Used to store contiguous time periods from each data source

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
                end_buffer = minutes(nwp_config.time_resolution_minutes)
            else:
                end_buffer = minutes(0)
            
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

    if "gsp" in datasets_dict:
        gsp_config = config.input_data.gsp

        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(datasets_dict["gsp"]["time_utc"]),
            sample_period_duration=minutes(gsp_config.time_resolution_minutes),
            history_duration=minutes(gsp_config.history_minutes),
            forecast_duration=minutes(gsp_config.forecast_minutes),
        )

        contiguous_time_periods['gsp'] = time_periods

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

    # Fill out the contiguous time periods to get the t0 times
    valid_t0_times = fill_time_periods(
        valid_time_periods, 
        freq=minutes(config.input_data.gsp.time_resolution_minutes)
    )

    return valid_t0_times


def get_gsp_locations(gsp_ids: list[int] | None = None) -> list[Location]:
    """Get list of locations of all GSPs"""
    
    if gsp_ids is None:
        gsp_ids = [i for i in range(1, 318)]
    
    locations = []

    # Load UK GSP locations
    df_gsp_loc = pd.read_csv(
        pkg_resources.resource_filename(__name__, "../data/uk_gsp_locations.csv"),
        index_col="gsp_id",
    )

    for gsp_id in gsp_ids:
        locations.append(
            Location(
                coordinate_system = "osgb",
                x=df_gsp_loc.loc[gsp_id].x_osgb,
                y=df_gsp_loc.loc[gsp_id].y_osgb,
                id=gsp_id,
            )
        )
    return locations



class PVNetUKRegionalDataset(Dataset):
    def __init__(
        self, 
        config_filename: str, 
        start_time: str | None = None,
        end_time: str| None = None,
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
        
        # Get t0 times where all input data is available
        valid_t0_times = find_valid_t0_times(datasets_dict, config)

        # Filter t0 times to given range
        if start_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times>=pd.Timestamp(start_time)]
            
        if end_time is not None:
            valid_t0_times = valid_t0_times[valid_t0_times<=pd.Timestamp(end_time)]

        # Construct list of locations to sample from
        locations = get_gsp_locations(gsp_ids)

        # Construct a lookup for locations - useful for users to construct sample by GSP ID
        location_lookup = {loc.id: loc for loc in locations}
        
        # Construct indices for sampling
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
        
        
    def __len__(self):
        return len(self.index_pairs)
    
    
    def _get_sample(self, t0: pd.Timestamp, location: Location) -> dict:
        """Generate the PVNet sample for given coordinates
        
        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = compute(sample_dict)

        sample = process_and_combine_datasets(sample_dict, self.config, t0, location)
        
        return sample
    
        
    def __getitem__(self, idx):
        
        # Get the coordinates of the sample
        t_index, loc_index = self.index_pairs[idx]
        location = self.locations[loc_index]
        t0 = self.valid_t0_times[t_index]
        
        # Generate the sample
        return self._get_sample(t0, location)
    

    def get_sample(self, t0: pd.Timestamp, gsp_id: int) -> dict:
        """Generate a sample for the given coordinates. 
        
        Useful for users to generate samples by GSP ID.
        
        Args:
            t0: init-time for sample
            gsp_id: GSP ID
        """
        # Check the user has asked for a sample which we have the data for
        assert t0 in self.valid_t0_times
        assert gsp_id in self.location_lookup

        location = self.location_lookup[gsp_id]
        
        return self._get_sample(t0, location)