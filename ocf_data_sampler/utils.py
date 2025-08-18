"""Miscellaneous helper functions."""

import pandas as pd
from xarray_tensorstore import read


def minutes(minutes: int | list[float]) -> pd.Timedelta | pd.TimedeltaIndex:
    """Timedelta minutes.

    Args:
        minutes: the number of minutes, single value or list
    """
    return pd.to_timedelta(minutes, unit="m")


def compute(xarray_dict: dict) -> dict:
    """Eagerly load a nested dictionary of xarray DataArrays."""
    
    # Load these keys first because they don't use tensorstore
    for key in ["gsp", "site"]:
        if key in xarray_dict:
            da = xarray_dict.pop(key)
            keys = [*xarray_dict.keys()]
            xarray_dict[key] = da.compute()
    else:
        keys = [*xarray_dict.keys()]

    for k in keys:
        v = xarray_dict[k]
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute()
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


def tensorstore_compute(xarray_dict: dict) -> dict:
    """Eagerly read and load a nested dictionary of xarray-tensorstore DataArrays."""
    return compute(tensorstore_read(xarray_dict))

