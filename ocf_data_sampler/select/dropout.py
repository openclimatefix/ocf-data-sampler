import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta, datetime


def draw_dropout_time(
        t0: datetime,
        dropout_timedeltas: list[timedelta] | None,
        dropout_frac: float = 0,
    ):

    if dropout_timedeltas is not None:
        assert len(dropout_timedeltas) >= 1, "Must include list of relative dropout timedeltas"
        assert all(
            [t < timedelta(minutes=0) for t in dropout_timedeltas]
        ), "dropout timedeltas must be negative"
    assert 0 <= dropout_frac <= 1

    if (dropout_timedeltas is None) or (np.random.uniform() >= dropout_frac):
        dropout_time = None
    else:
        t0_datetime_utc = pd.Timestamp(t0)
        dt = np.random.choice(dropout_timedeltas)
        dropout_time = t0_datetime_utc + dt

    return dropout_time


def apply_dropout_time(
        ds: xr.Dataset,
        dropout_time: pd.Timestamp | None,
    ):

    if dropout_time is None:
        return ds
    else:
        # This replaces the times after the dropout with NaNs
        return ds.where(ds.time_utc <= dropout_time)