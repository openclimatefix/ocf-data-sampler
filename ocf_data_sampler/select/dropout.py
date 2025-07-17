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
    dropout_frac: float|list[float],
    da: xr.DataArray,
) -> xr.DataArray:
    """Randomly pick a dropout time from a list of timedeltas and apply dropout time to the data.

    Args:
        t0: The forecast init-time
        dropout_timedeltas: List of timedeltas relative to t0 to pick from
        dropout_frac: Either a probability that dropout will be applied.
            This should be between 0 and 1 inclusive.
            Or a list of probabilities for each of the corresponding timedeltas
        da: Xarray DataArray with 'time_utc' coordinate
    """
    if  isinstance(dropout_frac, list):
        # checking if len match
        if len(dropout_frac) != len(dropout_timedeltas):
            raise ValueError("Lengths of dropout_frac and dropout_timedeltas should match")




        dropout_time = t0 + np.random.choice(dropout_timedeltas,p=dropout_frac)

        return da.where(da.time_utc <= dropout_time)



    # old logic
    else:
        # sample dropout time
        if dropout_frac > 0 and len(dropout_timedeltas) == 0:
            raise ValueError("To apply dropout, dropout_timedeltas must be provided")


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
