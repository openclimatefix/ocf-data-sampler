import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS
from ocf_data_sampler.numpy_batch import (
    convert_nwp_to_numpy_batch,
    convert_satellite_to_numpy_batch,
    convert_gsp_to_numpy_batch,
    make_sun_position_numpy_batch,
    convert_site_to_numpy_batch,
)
from ocf_data_sampler.numpy_batch.gsp import GSPBatchKey
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.select.geospatial import osgb_to_lon_lat
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.utils import minutes


def process_and_combine_datasets(
    dataset_dict: dict,
    config: Configuration,
    t0: pd.Timestamp,
    location: Location,
    target_key: str = 'gsp'
) -> dict:
    """Normalize and convert data to numpy arrays"""

    numpy_modalities = []

    if "nwp" in dataset_dict:

        nwp_numpy_modalities = dict()

        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            # Standardise
            provider = config.input_data.nwp[nwp_key].provider
            da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            # Convert to NumpyBatch
            nwp_numpy_modalities[nwp_key] = convert_nwp_to_numpy_batch(da_nwp)

        # Combine the NWPs into NumpyBatch
        numpy_modalities.append({NWPBatchKey.nwp: nwp_numpy_modalities})

    if "sat" in dataset_dict:
        # Satellite is already in the range [0-1] so no need to standardise
        da_sat = dataset_dict["sat"]

        # Convert to NumpyBatch
        numpy_modalities.append(convert_satellite_to_numpy_batch(da_sat))

    gsp_config = config.input_data.gsp

    if "gsp" in dataset_dict:
        da_gsp = xr.concat([dataset_dict["gsp"], dataset_dict["gsp_future"]], dim="time_utc")
        da_gsp = da_gsp / da_gsp.effective_capacity_mwp

        numpy_modalities.append(
            convert_gsp_to_numpy_batch(
                da_gsp, 
                t0_idx=-gsp_config.interval_start_minutes / gsp_config.time_resolution_minutes
            )
        )

        # Add coordinate data
        # TODO: Do we need all of these?
        numpy_modalities.append(
            {
                GSPBatchKey.gsp_id: location.id,
                GSPBatchKey.x_osgb: location.x,
                GSPBatchKey.y_osgb: location.y,
            }
        )


    if "site" in dataset_dict:
        site_config = config.input_data.site
        da_sites = dataset_dict["site"]
        da_sites = da_sites / da_sites.capacity_kwp

        numpy_modalities.append(
            convert_site_to_numpy_batch(
                da_sites, t0_idx=-site_config.interval_start_minutes / site_config.time_resolution_minutes
            )
        )

    if target_key == 'gsp':
        # Make sun coords NumpyBatch
        datetimes = pd.date_range(
            t0+minutes(gsp_config.interval_start_minutes),
            t0+minutes(gsp_config.interval_end_minutes),
            freq=minutes(gsp_config.time_resolution_minutes),
        )

        lon, lat = osgb_to_lon_lat(location.x, location.y)

    elif target_key == 'site':
        # Make sun coords NumpyBatch
        datetimes = pd.date_range(
            t0+minutes(site_config.interval_start_minutes),
            t0+minutes(site_config.interval_end_minutes),
            freq=minutes(site_config.time_resolution_minutes),
        )

        lon, lat = location.x, location.y

    numpy_modalities.append(
        make_sun_position_numpy_batch(datetimes, lon, lat, key_prefix=target_key)
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


def fill_nans_in_arrays(batch: dict) -> dict:
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


def compute(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays"""
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute(scheduler="single-threaded")
    return xarray_dict
