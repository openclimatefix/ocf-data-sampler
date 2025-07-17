"""Miscellaneous helper functions."""

import pandas as pd


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
            xarray_dict[k] = v.compute(scheduler="single-threaded")
    return xarray_dict
