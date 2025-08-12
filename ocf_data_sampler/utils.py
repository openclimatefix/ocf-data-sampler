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
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = compute(v)
        else:
            xarray_dict[k] = v.compute()
    return xarray_dict


def tensorstore_compute(xarray_dict: dict) -> dict:
    """Eagerly read and load a nested dictionary of xarray-tensorstore DataArrays."""
    # Kick off the tensorstore async reading
    for k, v in xarray_dict.items():
        if isinstance(v, dict):
            xarray_dict[k] = tensorstore_compute(v)
        else:
            xarray_dict[k] = read(v)

    # Running the compute function will wait until all arrays have been read
    return compute(xarray_dict)

