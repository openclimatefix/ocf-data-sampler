"""Functions for simulating dropout in time series data.

This is used for the following types of data: GSP, Satellite and Site
This is not used for NWP
"""

import numpy as np
import pandas as pd
import xarray as xr


def apply_history_dropout(
    t0: pd.Timestamp,
    dropout_timedeltas: list[pd.Timedelta],
    dropout_frac: float | list[float],
    da: xr.DataArray,
) -> xr.DataArray:
    """Apply randomly sampled dropout to the historical part of some sequence data.

    Dropped out data is replaced with NaNs

    Args:
        t0: The forecast init-time.
        dropout_timedeltas: List of timedeltas relative to t0 to pick from
        dropout_frac: The probabilit(ies) that each dropout timedelta will be applied. This should
            be between 0 and 1 inclusive.
        da: Xarray DataArray with 'time_utc' coordinate
    """
    if len(dropout_timedeltas)==0:
        return da

    if isinstance(dropout_frac, float | int):

        if not (0<=dropout_frac<=1):
            raise ValueError("`dropout_frac` must be in range [0, 1]")

        # Create list with equal chance for all dropout timedeltas
        n = len(dropout_timedeltas)
        dropout_frac = [dropout_frac/n for _ in range(n)]
    else:
        if not 0<=sum(dropout_frac)<=1:
            raise ValueError("The sum of `dropout_frac` must be in range [0, 1]")
        if len(dropout_timedeltas)!=len(dropout_frac):
            raise ValueError("`dropout_timedeltas` and `dropout_frac` must have the same length")

        dropout_frac = [*dropout_frac] # Make copy of the list so we can append to it

    dropout_timedeltas = [*dropout_timedeltas] # Make copy of the list so we can append to it

    # Add chance of no dropout
    dropout_frac.append(1-sum(dropout_frac))
    dropout_timedeltas.append(None)

    timedelta_choice = np.random.choice(dropout_timedeltas, p=dropout_frac)

    if timedelta_choice is None:
        return da
    else:
        return da.where((da.time_utc <= timedelta_choice + t0) | (da.time_utc> t0))
