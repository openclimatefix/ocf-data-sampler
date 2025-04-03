"""Functions for simulating dropout in time series data.

This is used for the following types of data: GSP, Satellite and Site
This is not used for NWP
"""

import numpy as np
import pandas as pd
import xarray as xr


def draw_dropout_time(
    t0: pd.Timestamp,
    dropout_timedeltas: list[pd.Timedelta],
    dropout_frac: float,
) -> pd.Timestamp:
    """Randomly pick a dropout time from a list of timedeltas.

    Args:
        t0: The forecast init-time
        dropout_timedeltas: List of timedeltas relative to t0 to pick from
        dropout_frac: Probability that dropout will be applied.
            This should be between 0 and 1 inclusive
    """
    if dropout_frac > 0 and len(dropout_timedeltas) == 0:
        raise ValueError("To apply dropout, dropout_timedeltas must be provided")

    for t in dropout_timedeltas:
        if t > pd.Timedelta("0min"):
            raise ValueError("Dropout timedeltas must be negative")

    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1 inclusive")

    if (len(dropout_timedeltas) == 0) or (np.random.uniform() >= dropout_frac):
        dropout_time = None
    else:
        dropout_time = t0 + np.random.choice(dropout_timedeltas)

    return dropout_time


def apply_dropout_time(
    ds: xr.DataArray,
    dropout_time: pd.Timestamp | None,
) -> xr.DataArray:
    """Apply dropout time to the data.

    Args:
        ds: Xarray DataArray with 'time_utc' coordinate
        dropout_time: Time after which data is set to NaN
    """
    if dropout_time is None:
        return ds
    else:
        # This replaces the times after the dropout with NaNs
        return ds.where(ds.time_utc <= dropout_time)
