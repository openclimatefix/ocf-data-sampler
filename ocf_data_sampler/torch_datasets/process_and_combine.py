import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS,RSS_MEAN,RSS_STD
from ocf_data_sampler.numpy_sample import (
    convert_nwp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    convert_gsp_to_numpy_sample,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from ocf_data_sampler.select.geospatial import osgb_to_lon_lat
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.utils import minutes


def process_and_combine_datasets(
    dataset_dict: dict,
    config: Configuration,
    t0: Optional[pd.Timestamp] = None,
    location: Optional[Location] = None,
    target_key: str = 'gsp'
) -> dict:

    """Normalise and convert data to numpy arrays"""
    numpy_modalities = []

    if "nwp" in dataset_dict:

        nwp_numpy_modalities = dict()

        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            # Standardise
            provider = config.input_data.nwp[nwp_key].provider
            da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            # Convert to NumpySample
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)

        # Combine the NWPs into NumpySample
        numpy_modalities.append({NWPSampleKey.nwp: nwp_numpy_modalities})


    if "sat" in dataset_dict:
        # Standardise
        da_sat = dataset_dict["sat"]
        da_sat = (da_sat - RSS_MEAN) / RSS_STD

        # Convert to NumpySample
        numpy_modalities.append(convert_satellite_to_numpy_sample(da_sat))


    gsp_config = config.input_data.gsp

    if "gsp" in dataset_dict:
        da_gsp = xr.concat([dataset_dict["gsp"], dataset_dict["gsp_future"]], dim="time_utc")
        da_gsp = da_gsp / da_gsp.effective_capacity_mwp

        numpy_modalities.append(
            convert_gsp_to_numpy_sample(
                da_gsp, 
                t0_idx=-gsp_config.interval_start_minutes / gsp_config.time_resolution_minutes
            )
        )

        # Add coordinate data
        # TODO: Do we need all of these?
        numpy_modalities.append(
            {
                GSPSampleKey.gsp_id: location.id,
                GSPSampleKey.x_osgb: location.x,
                GSPSampleKey.y_osgb: location.y,
            }
        )

    if target_key == 'gsp':
        # Make sun coords NumpySample
        datetimes = pd.date_range(
            t0+minutes(gsp_config.interval_start_minutes),
            t0+minutes(gsp_config.interval_end_minutes),
            freq=minutes(gsp_config.time_resolution_minutes),
        )

        lon, lat = osgb_to_lon_lat(location.x, location.y)

    numpy_modalities.append(
        make_sun_position_numpy_sample(datetimes, lon, lat, key_prefix=target_key)
    )

    # Combine all the modalities and fill NaNs
    combined_sample = merge_dicts(numpy_modalities)
    combined_sample = fill_nans_in_arrays(combined_sample)

    return combined_sample

def merge_dicts(list_of_dicts: list[dict]) -> dict:
    """Merge a list of dictionaries into a single dictionary"""
    # TODO: This doesn't account for duplicate keys, which will be overwritten
    combined_dict = {}
    for d in list_of_dicts:
        combined_dict.update(d)
    return combined_dict

def fill_nans_in_arrays(sample: dict) -> dict:
    """Fills all NaN values in each np.ndarray in the sample dictionary with zeros.

    Operation is performed in-place on the sample.
    """
    for k, v in sample.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if np.isnan(v).any():
                sample[k] = np.nan_to_num(v, copy=False, nan=0.0)

        # Recursion is included to reach NWP arrays in subdict
        elif isinstance(v, dict):
            fill_nans_in_arrays(v)

    return sample


def compute(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays"""
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute(scheduler="single-threaded")
    return xarray_dict
