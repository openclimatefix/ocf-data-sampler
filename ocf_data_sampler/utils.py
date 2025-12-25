"""Miscellaneous helper functions."""

import numpy as np
import pandas as pd
from xarray_tensorstore import read


def minutes(minutes: int | list[float]) -> pd.Timedelta | pd.TimedeltaIndex:
    """Timedelta minutes.

    Args:
        minutes: the number of minutes, single value or list
    """
    return pd.to_timedelta(minutes, unit="m")


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


def tensorstore_read(xarray_dict: dict) -> dict:
    """Start reading a nested dictionary of xarray-tensorstore DataArrays."""
    # Kick off the tensorstore async reading
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = tensorstore_read(v)
        else:
            xarray_dict[k] = read(v)
    return xarray_dict


def load_data_dict(xarray_dict: dict) -> dict:
    """Eagerly read and load a nested dictionary of xarray-tensorstore DataArrays."""
    return load(tensorstore_read(xarray_dict))

