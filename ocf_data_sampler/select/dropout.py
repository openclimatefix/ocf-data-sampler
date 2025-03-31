"""Function for simulating dropout in time series data.

This is used for the following types of data: GSP, Satellite and Site
This is not used for NWP
"""

import numpy as np
import pandas as pd
import xarray as xr

def simulate_dropout(
    ds: xr.DataArray,
    t0: pd.Timestamp,
    dropout_timedeltas: list[pd.Timedelta],
    dropout_frac: float,
) -> xr.DataArray:
    """
    Combined function that draws a dropout time and applies dropout.
    This function preserves the functionalities of the original two functions,
    `draw_dropout_time` and `apply_dropout_time`.

    Args:
        ds: Xarray DataArray with a 'time_utc' coordinate.
        t0: Forecast initialization time.
        dropout_timedeltas: List of negative timedeltas relative to t0.
        dropout_frac: Probability that dropout will be applied (between 0 and 1).

    Returns:
        A new DataArray with values after the chosen dropout time set to NaN.
    """
    # Validate inputs (same as in draw_dropout_time)
    if dropout_frac > 0 and len(dropout_timedeltas) == 0:
        raise ValueError("To apply dropout, dropout_timedeltas must be provided")

    for t in dropout_timedeltas:
        if t > pd.Timedelta("0min"):
            raise ValueError("Dropout timedeltas must be negative")

    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1 inclusive")

    # Determine dropout time (same logic as draw_dropout_time)
    if (len(dropout_timedeltas) == 0) or (np.random.uniform() >= dropout_frac):
        dropout_time = t0
    else:
        dropout_time = t0 + np.random.choice(dropout_timedeltas)

    # Apply dropout (same as apply_dropout_time)
    return ds.where(ds.time_utc <= dropout_time)