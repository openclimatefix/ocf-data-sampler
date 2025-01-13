import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS, RSS_MEAN, RSS_STD

def process_and_combine_site_sample_dict(
    dataset_dict: dict,
    config: Configuration,
) -> xr.Dataset:
    """
    Normalize and combine data into a single xr Dataset

    Args:
        dataset_dict: dict containing sliced xr DataArrays
        config: Configuration for the model

    Returns:
        xr.Dataset: A merged Dataset with nans filled in.
    
    """

    data_arrays = []

    if "nwp" in dataset_dict:
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            # Standardise
            provider = config.input_data.nwp[nwp_key].provider
            da_nwp = (da_nwp - NWP_MEANS[provider]) / NWP_STDS[provider]
            data_arrays.append((f"nwp-{provider}", da_nwp))
          
    if "sat" in dataset_dict:
        # Standardise
        da_sat = dataset_dict["sat"]
        da_sat = (da_sat - RSS_MEAN) / RSS_STD
        data_arrays.append(("satellite", da_sat))

    if "site" in dataset_dict:
        # site_config = config.input_data.site
        da_sites = dataset_dict["site"]
        da_sites = da_sites / da_sites.capacity_kwp
        data_arrays.append(("sites", da_sites))
    
    combined_sample_dataset = merge_arrays(data_arrays)

    # Fill any nan values
    return combined_sample_dataset.fillna(0.0)


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
