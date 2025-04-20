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
    """Simulate data dropout by masking values after a randomly chosen time offset.

    Args:
        ds: Input data with 'time_utc' coordinate
        t0: Reference time for calculating dropout offsets
        dropout_timedeltas: Time offsets (must be ≤ 0) or empty list for no dropout
        dropout_frac: Probability of applying dropout (0-1)

    Returns:
        DataArray with NaN values after the dropout time (if applied)
    """
    # Validate input parameters in correct order
    if not 0 <= dropout_frac <= 1:
        raise ValueError("dropout_frac must be between 0 and 1")

    if any(t > pd.Timedelta(0) for t in dropout_timedeltas):
        raise ValueError("All dropout offsets must be ≤ 0")

    if dropout_frac > 0 and len(dropout_timedeltas) == 0:
        raise ValueError("Must provide dropout_timedeltas when dropout_frac > 0")

    # Early return if no dropout
    if len(dropout_timedeltas) == 0 or np.random.uniform() >= dropout_frac:
        return ds.copy()

    # Apply dropout
    dropout_time = t0 + np.random.choice(dropout_timedeltas)
    return ds.where(ds.time_utc <= dropout_time)
