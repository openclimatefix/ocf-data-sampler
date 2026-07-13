"""Utility function for filling NaNs in DataArrays."""

import numpy as np
import xarray as xr

from ocf_data_sampler.config.model import Configuration, DropoutMixin


def fill_nans(da: xr.DataArray, source_config: DropoutMixin) -> xr.DataArray:
    """Fill NaNs in a DataArray in-place."""
    if np.isnan(da.data).any():
        da.data = np.nan_to_num(da.data, copy=False, nan=source_config.dropout_value)
    return da


def fill_nans_in_dataset_dicts(datasets_dict: dict, config: Configuration) -> dict:
    """Fills all NaN values in the dataarrays in-place.

    Args:
        datasets_dict: Dictionary of the input data sources
        config: Configuration object.
    """
    conf_in = config.input_data
    if "generation" in datasets_dict:
        datasets_dict["generation"] = fill_nans(datasets_dict["generation"], conf_in.generation)

    if "sat" in datasets_dict:
        datasets_dict["sat"] = fill_nans(datasets_dict["sat"], conf_in.satellite)

    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.input_data.nwp.items():
            datasets_dict["nwp"][nwp_key] = fill_nans(datasets_dict["nwp"][nwp_key], nwp_config)

    return datasets_dict
