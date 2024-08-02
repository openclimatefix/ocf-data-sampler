"""Torch dataset for PVNet"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset


from ocf_data_sampler.load.gsp import open_gsp
from ocf_data_sampler.load.nwp import open_nwp
from ocf_data_sampler.load.satellite import open_sat_data
from ocf_data_sampler.select.find_contiguous_t0_time_periods import (
    find_contiguous_t0_time_periods, find_contiguous_t0_periods_nwp, 
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
    add_sun_position,
)


from ocf_datapipes.config.model import Configuration
from ocf_datapipes.config.load import load_yaml_configuration

from ocf_datapipes.utils.location import Location

from ocf_datapipes.batch import BatchKey

from ocf_datapipes.utils.consts import (
    NWP_MEANS,
    NWP_STDS,
    RSS_MEAN,
    RSS_STD,
)

from ocf_datapipes.training.common import (
    is_config_and_path_valid, minutes_list_to_timedeltas, concat_xr_time_utc, normalize_gsp,
)


xr.set_options(keep_attrs=True)


xarray_object = xr.DataArray | xr.Dataset

# TODO: This is messy. Remove the need for this in NumpyBatch conversion
def add_t0_idx_and_sample_period_duration(ds: xarray_object, source_config):
    """Add attributes to the xarray dataset needed for numpy batch

    Args:
        ds: Xarray object
        source_config: Input data source config
    """

    ds.attrs["t0_idx"] = int(
        source_config.history_minutes / source_config.time_resolution_minutes
    )
    ds.attrs["sample_period_duration"] = minutes(source_config.time_resolution_minutes)
    return ds


def minutes(m: float):
    """Timedelta minutes
    
    Args:
        m: minutes
    """
    return timedelta(minutes=m)


def get_dataset_dict(config: Configuration):
    """Construct dictionary of all of the input data sources

    Args:
        config: Configuration file
    """
    # Check which modalities to use
    conf_in = config.input_data
    use_nwp = (
        
        (conf_in.nwp is not None)
        and len(conf_in.nwp) != 0
        and all(v.nwp_zarr_path != "" for _, v in conf_in.nwp.items())
    )
    use_sat = is_config_and_path_valid(True, conf_in.satellite, "satellite_zarr_path")
    use_gsp = is_config_and_path_valid(True, conf_in.gsp, "gsp_zarr_path")


    datasets = {}

    # We always assume GSP will be included
    gsp_config = config.input_data.gsps

    ds_gsp = open_gsp(zarr_path=gsp_config.gsp_zarr_path)
    ds_gsp = add_t0_idx_and_sample_period_duration(ds_gsp, gsp_config)

    datasets["gsp"] = ds_gsp

    # Load NWP data if in config
    if use_nwp:
        
        datasets["nwp"] = {}
        for nwp_source, nwp_config in conf_in.nwp.items():

            ds_nwp = open_nwp(nwp_config.nwp_zarr_path, provider=nwp_config.nwp_provider)

            ds_nwp = ds_nwp.sel(channel=list(nwp_config.nwp_channels))

            ds_nwp = add_t0_idx_and_sample_period_duration(ds_nwp, nwp_config)

            datasets["nwp"][nwp_source] = ds_nwp

    # Load satellite data if in config
    if use_sat:
        sat_config = config.input_data.satellite

        ds_sat = open_sat_data(sat_config.satellite_zarr_path)

        ds_sat.sel(channel=list(sat_config.satellite_channels))

        ds_sat = add_t0_idx_and_sample_period_duration(ds_sat, sat_config)

        datasets["sat"] = ds_sat

    return datasets

    
    
def find_valid_t0_times(
    datasets_dict: dict,
    config: Configuration,
):
    """Find the t0 times where all of the requested input data is available

    Args:
        datasets_dict: A dictionary of input datasets 
        config: Configuration file
    """

    contiguous_time_periods = []  # Used to store contiguous time periods from each data source

    # TODO: Is this cleaner as series of `if key in datasets_dict` statements rather than  loop?
    for key in datasets_dict.keys():

        if key == "nwp":
            for nwp_key, nwp_conf in config.input_data.nwp.items():

                ds = datasets_dict["nwp"][nwp_key]

                if nwp_conf.dropout_timedeltas_minutes is None:
                    max_dropout = minutes(0)
                else:
                    max_dropout = minutes(int(np.max(np.abs(nwp_conf.dropout_timedeltas_minutes))))

                if nwp_conf.max_staleness_minutes is None:
                    max_staleness = None
                else:
                    max_staleness = minutes(nwp_conf.max_staleness_minutes)

                # The last step of the forecast is lost if we have to diff channels
                if len(nwp_conf.nwp_accum_channels) > 0:
                    # TODO: this hard codes the assumption of hourly steps
                    end_buffer = minutes(60)
                else:
                    end_buffer = minutes(0)
                
                # This is the max staleness we can use considering the max step of the input data
                max_possible_staleness = (
                    pd.Timedelta(ds["step"].max().item())
                    - minutes(nwp_conf.forecast_minutes)
                    - end_buffer
                )

                # Default to use max possible staleness unless specified in config
                if max_staleness is None:
                    max_staleness = max_possible_staleness
                else:
                    # Make sure the max acceptable staleness isn't longer than the max possible
                    assert max_staleness <= max_possible_staleness
                    max_staleness = max_staleness

                time_periods = find_contiguous_t0_periods_nwp(
                    datetimes=pd.DatetimeIndex(ds["init_time_utc"]),
                    history_duration=minutes(nwp_conf.history_minutes),
                    max_staleness=max_staleness,
                    max_dropout=max_dropout,
                )

                contiguous_time_periods.append(time_periods)

        else:
            if key == "sat":
                key_config = config.input_data.satellite
            elif key == "gsp":
                key_config = config.input_data.gsp
            else:
                raise ValueError(f"Unexpected key: {key}")


            time_periods = find_contiguous_t0_time_periods(
                pd.DatetimeIndex(datasets_dict[key]["time_utc"]),
                sample_period_duration=minutes(key_config.time_resolution_minutes),
                history_duration=minutes(key_config.history_minutes),
                forecast_duration=minutes(key_config.forecast_minutes),
            )

            contiguous_time_periods.append(time_periods)

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_periods) > 1:
        valid_time_periods = intersection_of_multiple_dataframes_of_periods(
            contiguous_time_periods
        )
    else:
        valid_time_periods = contiguous_time_periods[0]

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

    sliced_datasets_dict = {}
    
    if "nwp" in datasets_dict:

        sliced_datasets_dict["nwp"] = {}
        
        for nwp_key, nwp_config in config.input_data.nwp.items():

            sliced_datasets_dict["nwp"][nwp_key] = select_spatial_slice_pixels(
                datasets_dict["nwp"][nwp_key],
                location,
                roi_height_pixels=nwp_config.nwp_image_size_pixels_height,
                roi_width_pixels=nwp_config.nwp_image_size_pixels_width,
            )

    if "sat" in datasets_dict:
        conf_sat = config.input_data.satellite

        sliced_datasets_dict["sat"] = select_spatial_slice_pixels(
            datasets_dict["sat"],
            location,
            roi_height_pixels=conf_sat.satellite_image_size_pixels_height,
            roi_width_pixels=conf_sat.satellite_image_size_pixels_width,
        )

    # GSP always assumed to be in data
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
    conf_in = config.input_data

    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        
        sliced_datasets_dict["nwp"] = {}
        
        for nwp_key, ds_nwp in datasets_dict["nwp"].items():
            
            dropout_timedeltas = minutes_list_to_timedeltas(
                conf_in.nwp[nwp_key].dropout_timedeltas_minutes
            )
                
            sliced_datasets_dict["nwp"][nwp_key] = select_time_slice_nwp(
                ds_nwp,
                t0,
                sample_period_duration=minutes(conf_in.nwp[nwp_key].time_resolution_minutes),
                history_duration=minutes(conf_in.nwp[nwp_key].history_minutes),
                forecast_duration=minutes(conf_in.nwp[nwp_key].forecast_minutes),
                dropout_timedeltas=dropout_timedeltas,
                dropout_frac=conf_in.nwp[nwp_key].dropout_fraction,
                accum_channels=conf_in.nwp[nwp_key].nwp_accum_channels,
            )

    if "sat" in datasets_dict:

        sliced_datasets_dict["sat"] = select_time_slice(
            datasets_dict["sat"],
            t0,
            sample_period_duration=minutes(conf_in.satellite.time_resolution_minutes),
            interval_start=minutes(-conf_in.satellite.history_minutes),
            interval_end=minutes(-conf_in.satellite.live_delay_minutes),
            max_steps_gap=2,
        )

        # Randomly sample dropout
        sat_dropout_time = draw_dropout_time(
            t0,
            dropout_timedeltas=minutes_list_to_timedeltas(
                conf_in.satellite.dropout_timedeltas_minutes
            ),
            dropout_frac=conf_in.satellite.dropout_fraction,
        )

        # Apply the dropout
        sliced_datasets_dict["sat"] = apply_dropout_time(
            sliced_datasets_dict["sat"],
            sat_dropout_time,
        )

    # GSP always assumed to be included
    sliced_datasets_dict["gsp_future"] = select_time_slice(
        datasets_dict["gsp"],
        t0,
        sample_period_duration=minutes(conf_in.gsp.time_resolution_minutes),
        interval_start=minutes(30),
        interval_end=minutes(conf_in.gsp.forecast_minutes),
    )
        
    sliced_datasets_dict["gsp"] = select_time_slice(
        datasets_dict["gsp"],
        t0,
        sample_period_duration=minutes(conf_in.gsp.time_resolution_minutes),
        interval_start=-minutes(conf_in.gsp.history_minutes),
        interval_end=minutes(0),
    )

    # Dropout on the GSP, but not the future GSP
    dropout_timedeltas = minutes_list_to_timedeltas(conf_in.gsp.dropout_timedeltas_minutes)

    gsp_dropout_time = draw_dropout_time(
        t0,
        dropout_timedeltas=dropout_timedeltas,
        dropout_frac=conf_in.gsp.dropout_fraction,
    )

    sliced_datasets_dict["gsp"] = apply_dropout_time(
        sliced_datasets_dict["gsp"],
        gsp_dropout_time,
    )


    return sliced_datasets_dict


def merge_dicts(list_of_dicts: list[dict]):
    """Merge a list of dictionaries into a single dictionary"""
    # TODO: This doesn't account for duplicate keys
    all_d = {}
    for d in list_of_dicts:
        all_d.update(d)
    return all_d



def process_and_combine_datasets(dataset_dict: dict, config: Configuration):
    """Normalize and convert data to numpy arrays"""    

    numpy_modalities = []

    if "nwp" in dataset_dict:

        conf_nwp = config.input_data.nwp
        nwp_numpy_modalities = dict()

        for nwp_key, ds_nwp in dataset_dict["nwp"].items():
            # Normalise
            provider = conf_nwp[nwp_key].nwp_provider
            ds_nwp = (ds_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            # Convert to NumpyBatch
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_batch(ds_nwp)
        
        # Combine the NWPs into NumpyBatch
        numpy_modalities.append({BatchKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        # Normalise
        ds_sat = (dataset_dict["sat"] - RSS_MEAN) / RSS_STD
        # Convert to NumpyBatch
        numpy_modalities.append(convert_satellite_to_numpy_batch(ds_sat))


    # GSP always assumed to be in data
    ds_gsp = concat_xr_time_utc([dataset_dict["gsp"], dataset_dict["gsp_future"]])
    ds_gsp = normalize_gsp(ds_gsp)
    numpy_modalities.append(convert_gsp_to_numpy_batch(ds_gsp))

    # Combine all the modalities
    combined_sample = merge_dicts(numpy_modalities)

    # Add sun coords
    combined_sample = add_sun_position(combined_sample, modality_name="gsp")
 
    return combined_sample


def compute(d):
    """Eagerly load a nested dictionary of xarray data"""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = compute(v)
        else:
            d[k] = v.compute(scheduler="single-threaded")
    return d


def get_locations(gs_gsp: xr.Dataset) -> list[Location]:
    """Get list of locations of GSP"""
    locations = []
    for gsp_id in gs_gsp.gsp_id.values:
        ds_ = gs_gsp.sel(gsp_id=gsp_id)
        locations.append(
            Location(
                coordinate_system = "osgb",
                x=ds_.x_osgb.item(),
                y=ds_.y_osgb.item(),
                id=gsp_id,
            )
        )
    return locations


class PVNetDataset(Dataset):
    def __init__(
        self, 
        config_filename: str, 
        start_time: str | None = None,
        end_time: str| None = None,
    ):
        """A torch Dataset for PVNet

        """
        
        config = load_yaml_configuration(config_filename)
        
        datasets_dict = get_dataset_dict(config)

        # Remove national GSP ID
        datasets_dict["gsp"] = datasets_dict["gsp"].sel(gsp_id=slice(1, None))

        if (start_time is not None) or (end_time is not None):
            datasets_dict["gsp"] = datasets_dict["gsp"].sel(time_utc=slice(start_time, end_time))
        
        # Get t0 times where all input data is available
        valid_t0_times = find_valid_t0_times(
            datasets_dict,
            config,
        )

        # Construct list of locations to sample from
        locations = get_locations(datasets_dict["gsp"])
        
        # Construct indices for sampling
        t_index, loc_index = np.meshgrid(
            np.arange(len(valid_t0_times)),
            np.arange(len(locations)),
        )

        index_pairs = np.stack((t_index.ravel(), loc_index.ravel())).T
            
        # Assign coords and indices to self
        self.valid_t0_times = valid_t0_times
        self.locations = locations
        self.index_pairs = index_pairs

        # Assign config and input data to self
        self.datasets_dict = datasets_dict
        self.config = config
        
        
    def __len__(self):
        return len(self.index_pairs)
    
    # TODO: Would this be better if we could pass in GSP ID int instead of Location object?
    def get_sample(self, t0: pd.Timestamp, location: Location):
        """Generate the PVNet sample for given coordinates
        
        Args:
            t0: init-time for sample
            location: location for sample
        """
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        sample_dict = compute(sample_dict)

        sample = process_and_combine_datasets(sample_dict, self.config)
        
        return sample
    
        
    def __getitem__(self, idx):
        
        # Get the coordinates of the sample
        t_index, loc_index = self.index_pairs[idx]
        location = self.locations[loc_index]
        t0 = self.valid_t0_times[t_index]
        
        # Generate the sample
        return self.get_sample(t0, location)
    
    
if __name__=="__main__":

    # ------------------ basic usage ---------------------

    # TODO: remove this, its messy, but useful until we have tests and docs
    config_filename = (
        "/home/jamesfulton/repos/PVNet/configs/datamodule/configuration/gcp_configuration.yaml"
    )

    # Create dataset object
    dataset = PVNetDataset(config_filename)
    
    # Print number of samples
    print(f"Found {len(dataset.valid_t0_times)} possible samples")

    # Find the 0th sample coordinates
    # TODO: Should we be able to use the dataset to map from index to t0, location coords more 
    # easily?
    idx = 0
    t_index, loc_index = dataset.index_pairs[idx]

    location = dataset.locations[loc_index]
    t0 = dataset.valid_t0_times[t_index]

    # Print coords
    print(t0)
    print(location)
    
    # Generate sample - no printing since its BIG
    sample = dataset[idx]
