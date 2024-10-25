import numpy as np
import pandas as pd
import xarray as xr


def draw_dropout_time(
        t0: pd.Timestamp,
        dropout_timedeltas: list[pd.Timedelta] | None,
        dropout_frac: float = 0,
    ):

    if dropout_timedeltas is not None:
        assert len(dropout_timedeltas) >= 1, "Must include list of relative dropout timedeltas"
        assert all(
            [t <= pd.Timedelta("0min") for t in dropout_timedeltas]
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
        ds: xr.DataArray,
        dropout_time: pd.Timestamp | None,
    ):

    if dropout_time is None:
        return ds
    else:
        # This replaces the times after the dropout with NaNs
        return ds.where(ds.time_utc <= dropout_time)
