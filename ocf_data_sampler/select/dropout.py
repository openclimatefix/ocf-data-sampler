"""Functions for simulating dropout in time series data.

This is used for the following types of data: GSP, Satellite and Site
This is not used for NWP
"""

import numpy as np
import pandas as pd
import xarray as xr


def apply_sampled_dropout_time(
    t0: pd.Timestamp,
    dropout_timedeltas: list[pd.Timedelta],
    dropout_frac: float,
    da: xr.DataArray,
) -> xr.DataArray:
    """Randomly pick a dropout time from a list of timedeltas and apply dropout time to the data.

    Args:
        t0: The forecast init-time
        dropout_timedeltas: List of timedeltas relative to t0 to pick from
        dropout_frac: Probability that dropout will be applied.
            This should be between 0 and 1 inclusive
        da: Xarray DataArray with 'time_utc' coordinate
    """
    # sample dropout time
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

    # apply dropout time
    if dropout_time is None:
        return da
    # This replaces the times after the dropout with NaNs
    return da.where(da.time_utc <= dropout_time)
