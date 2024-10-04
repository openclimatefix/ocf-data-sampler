"""Torch dataset for PVNet"""

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import pkg_resources

from ocf_data_sampler.load.gsp import open_gsp
from ocf_data_sampler.load.nwp import open_nwp
from ocf_data_sampler.load.satellite import open_sat_data

from ocf_data_sampler.select.find_contiguous_time_periods import (
    find_contiguous_t0_periods, find_contiguous_t0_periods_nwp, 
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.select.fill_time_periods import fill_time_periods
from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp
from ocf_data_sampler.select.dropout import draw_dropout_time, apply_dropout_time
from ocf_data_sampler.select.select_spatial_slice import select_spatial_slice_pixels

from ocf_data_sampler.numpy_batch import (
    convert_gsp_to_numpy_batch,
    convert_nwp_to_numpy_batch,
    convert_satellite_to_numpy_batch,
    make_sun_position_numpy_batch,
)


from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_datapipes.batch import BatchKey, NumpyBatch

from ocf_datapipes.utils.location import Location
from ocf_datapipes.utils.geospatial import osgb_to_lon_lat

from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
)

from ocf_datapipes.training.common import concat_xr_time_utc, normalize_gsp



xr.set_options(keep_attrs=True)



def minutes(minutes: list[float]):
    """Timedelta minutes
    
    Args:
        m: minutes
    """
    return pd.to_timedelta(minutes, unit="m")


def get_dataset_dict(config: Configuration) -> dict[xr.DataArray, dict[xr.DataArray]]:
    """Construct dictionary of all of the input data sources

    Args:
        config: Configuration file
    """
    
    in_config = config.input_data

    datasets_dict = {}

    # Load GSP data unless the path is None
    if in_config.gsp.gsp_zarr_path:
        da_gsp = open_gsp(zarr_path=in_config.gsp.gsp_zarr_path).compute()

        # Remove national GSP
        datasets_dict["gsp"] = da_gsp.sel(gsp_id=slice(1, None))

    # Load NWP data if in config
    if in_config.nwp:
        
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in in_config.nwp.items():

            da_nwp = open_nwp(nwp_config.nwp_zarr_path, provider=nwp_config.nwp_provider)

            da_nwp = da_nwp.sel(channel=list(nwp_config.nwp_channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if in_config.satellite:
        sat_config = config.input_data.satellite

        da_sat = open_sat_data(sat_config.satellite_zarr_path)

        da_sat = da_sat.sel(channel=list(sat_config.satellite_channels))

        datasets_dict["sat"] = da_sat

    return datasets_dict

    
    
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


def slice_datasets_by_space(
    datasets_dict: dict,
    location: Location,
    config: Configuration,
) -> dict:
    """Slice a dictionaries of input data sources around a given location

    Args:
        datasets_dict: Dictionary of the input data sources
        location: The location to sample around
        config: Configuration object.
    """

    assert set(datasets_dict.keys()).issubset({"nwp", "sat", "gsp"})

    sliced_datasets_dict = {}
    
    if "nwp" in datasets_dict:

        sliced_datasets_dict["nwp"] = {}
        
        for nwp_key, nwp_config in config.input_data.nwp.items():

            sliced_datasets_dict["nwp"][nwp_key] = select_spatial_slice_pixels(
                datasets_dict["nwp"][nwp_key],
                location,
                height_pixels=nwp_config.nwp_image_size_pixels_height,
                width_pixels=nwp_config.nwp_image_size_pixels_width,
            )

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        sliced_datasets_dict["sat"] = select_spatial_slice_pixels(
            datasets_dict["sat"],
            location,
            height_pixels=sat_config.satellite_image_size_pixels_height,
            width_pixels=sat_config.satellite_image_size_pixels_width,
        )

    if "gsp" in datasets_dict:
        sliced_datasets_dict["gsp"] = datasets_dict["gsp"].sel(gsp_id=location.id)

    return sliced_datasets_dict
    
    
def slice_datasets_by_time(
    datasets_dict: dict,
    t0: pd.Timedelta,
    config: Configuration,
) -> dict:
    """Slice a dictionaries of input data sources around a given t0 time

    Args:
        datasets_dict: Dictionary of the input data sources
        t0: The init-time
        config: Configuration object.
    """

    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        
        sliced_datasets_dict["nwp"] = {}
        
        for nwp_key, da_nwp in datasets_dict["nwp"].items():
                            
            nwp_config = config.input_data.nwp[nwp_key]

            sliced_datasets_dict["nwp"][nwp_key] = select_time_slice_nwp(
                da_nwp,
                t0,
                sample_period_duration=minutes(nwp_config.time_resolution_minutes),
                history_duration=minutes(nwp_config.history_minutes),
                forecast_duration=minutes(nwp_config.forecast_minutes),
                dropout_timedeltas=minutes(nwp_config.dropout_timedeltas_minutes),
                dropout_frac=nwp_config.dropout_fraction,
                accum_channels=nwp_config.nwp_accum_channels,
            )

    if "sat" in datasets_dict:

        sat_config = config.input_data.satellite

        sliced_datasets_dict["sat"] = select_time_slice(
            datasets_dict["sat"],
            t0,
            sample_period_duration=minutes(sat_config.time_resolution_minutes),
            interval_start=minutes(-sat_config.history_minutes),
            interval_end=minutes(-sat_config.live_delay_minutes),
            max_steps_gap=2,
        )

        # Randomly sample dropout
        sat_dropout_time = draw_dropout_time(
            t0,
            dropout_timedeltas=minutes(sat_config.dropout_timedeltas_minutes),
            dropout_frac=sat_config.dropout_fraction,
        )

        # Apply the dropout
        sliced_datasets_dict["sat"] = apply_dropout_time(
            sliced_datasets_dict["sat"],
            sat_dropout_time,
        )

    if "gsp" in datasets_dict:
        gsp_config = config.input_data.gsp

        sliced_datasets_dict["gsp_future"] = select_time_slice(
            datasets_dict["gsp"],
            t0,
            sample_period_duration=minutes(gsp_config.time_resolution_minutes),
            interval_start=minutes(30),
            interval_end=minutes(gsp_config.forecast_minutes),
        )
            
        sliced_datasets_dict["gsp"] = select_time_slice(
            datasets_dict["gsp"],
            t0,
            sample_period_duration=minutes(gsp_config.time_resolution_minutes),
            interval_start=-minutes(gsp_config.history_minutes),
            interval_end=minutes(0),
        )

        # Dropout on the GSP, but not the future GSP
        gsp_dropout_time = draw_dropout_time(
            t0,
            dropout_timedeltas=minutes(gsp_config.dropout_timedeltas_minutes),
            dropout_frac=gsp_config.dropout_fraction,
        )

        sliced_datasets_dict["gsp"] = apply_dropout_time(sliced_datasets_dict["gsp"], gsp_dropout_time)

    return sliced_datasets_dict


def fill_nans_in_arrays(batch: NumpyBatch) -> NumpyBatch:
    """Fills all NaN values in each np.ndarray in the batch dictionary with zeros.

    Operation is performed in-place on the batch.
    """
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if np.isnan(v).any():
                batch[k] = np.nan_to_num(v, copy=False, nan=0.0)

        # Recursion is included to reach NWP arrays in subdict
        elif isinstance(v, dict):
            fill_nans_in_arrays(v)

    return batch



def merge_dicts(list_of_dicts: list[dict]) -> dict:
    """Merge a list of dictionaries into a single dictionary"""
    # TODO: This doesn't account for duplicate keys, which will be overwritten
    combined_dict = {}
    for d in list_of_dicts:
        combined_dict.update(d)
    return combined_dict


def process_and_combine_datasets(
        dataset_dict: dict, 
        config: Configuration,
        t0: pd.Timedelta,
        location: Location,
    ) -> NumpyBatch:
    """Normalize and convert data to numpy arrays"""    

    numpy_modalities = []

    if "nwp" in dataset_dict:

        nwp_numpy_modalities = dict()

        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            # Standardise
            provider = config.input_data.nwp[nwp_key].nwp_provider
            da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            # Convert to NumpyBatch
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_batch(da_nwp)
        
        # Combine the NWPs into NumpyBatch
        numpy_modalities.append({BatchKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        # Satellite is already in the range [0-1] so no need to standardise
        da_sat = dataset_dict["sat"]

        # Convert to NumpyBatch
        numpy_modalities.append(convert_satellite_to_numpy_batch(da_sat))

    gsp_config = config.input_data.gsp
    
    if "gsp" in dataset_dict:
        da_gsp = concat_xr_time_utc([dataset_dict["gsp"], dataset_dict["gsp_future"]])
        da_gsp = normalize_gsp(da_gsp)

        numpy_modalities.append(
            convert_gsp_to_numpy_batch(
                da_gsp, 
                t0_idx=gsp_config.history_minutes / gsp_config.time_resolution_minutes
            )
        )

    # Make sun coords NumpyBatch
    datetimes = pd.date_range(
        t0-minutes(gsp_config.history_minutes),
        t0+minutes(gsp_config.forecast_minutes),
        freq=minutes(gsp_config.time_resolution_minutes),
    )

    lon, lat = osgb_to_lon_lat(location.x, location.y)

    numpy_modalities.append(make_sun_position_numpy_batch(datetimes, lon, lat))
    
    # Add coordinate data
    # TODO: Do we need all of these?
    numpy_modalities.append({
        BatchKey.gsp_id: location.id,
        BatchKey.gsp_x_osgb: location.x,
        BatchKey.gsp_y_osgb: location.y,
    })

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    combined_sample = fill_nans_in_arrays(combined_sample)

    return combined_sample


def compute(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays"""
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute(scheduler="single-threaded")
    return xarray_dict


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
    
    
    def _get_sample(self, t0: pd.Timestamp, location: Location) -> NumpyBatch:
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
    

    def get_sample(self, t0: pd.Timestamp, gsp_id: int) -> NumpyBatch:
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