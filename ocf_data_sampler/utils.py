"""Miscellaneous helper functions."""

import numpy as np
import pandas as pd
from xarray_tensorstore import read as xtr_read
from ocf_data_sampler.torch_datasets.fastarray import FastDataArray


def minutes(minutes: int | list[float]) -> pd.Timedelta | pd.TimedeltaIndex:
    """Timedelta minutes.

    Args:
        minutes: the number of minutes, single value or list
    """
    if isinstance(minutes, list):
        return np.array(minutes, dtype="timedelta64[m]")
    else:
        return np.timedelta64(minutes, "m")


def load(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays."""
    # Check the generation data is loaded
    if "generation" in xarray_dict and not isinstance(xarray_dict["generation"].data, np.ndarray):
        raise ValueError("Generation data is expected to already be loaded")

    # Load the rest
    keys = [k for k in xarray_dict if k != "generation"]
    for k in keys:
        v = xarray_dict[k]
        if isinstance(v, dict):
            xarray_dict[k] = load(v)
        else:
            xarray_dict[k] = v.load()
    return xarray_dict


def read_data_dict(xarray_dict: dict) -> dict:
    """Start reading a nested dictionary of DataArrays."""
    # Kick off the tensorstore async reading
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = read_data_dict(v)
        else:
            if isinstance(v, FastDataArray):
                xarray_dict[k].read()
            else:
                xarray_dict[k] = xtr_read(v)
    return xarray_dict


def load_data_dict(xarray_dict: dict) -> dict:
    """Eagerly read and load a nested dictionary of DataArrays."""
    return load(read_data_dict(xarray_dict))

