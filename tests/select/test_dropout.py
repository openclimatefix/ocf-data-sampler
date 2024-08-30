from ocf_data_sampler.select.dropout import draw_dropout_time, apply_dropout_time

import numpy as np
import pandas as pd
import xarray as xr

import pytest


@pytest.fixture(scope="module")
def da_sample():
    """Create dummy data which looks like satellite data"""

    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords=dict(
            time_utc=(["time_utc"], datetimes),
        )
    )
    return da_sat


def test_draw_dropout_time():
    t0 = pd.Timestamp("2021-01-01 04:00:00")

    dropout_timedeltas = pd.to_timedelta([-30, -60], unit="min")
    dropout_time = draw_dropout_time(t0, dropout_timedeltas, dropout_frac=1)

    assert isinstance(dropout_time, pd.Timestamp)
    assert dropout_time-t0 in dropout_timedeltas


def test_draw_dropout_time_partial():
    t0 = pd.Timestamp("2021-01-01 04:00:00")

    dropout_timedeltas = pd.to_timedelta([-30, -60], unit="min")

    dropouts = set()

    # Loop over 1000 to have very high probability of seeing all dropouts
    # The chances of this failing by chance are approx ((2/3)^100)*3 = 7e-18
    for _ in range(100):
        dropouts.add(draw_dropout_time(t0, dropout_timedeltas, dropout_frac=2/3))

    # Check all expected dropouts are present
    dropouts == {None} | set(t0 + dt for dt in dropout_timedeltas)


def test_draw_dropout_time_none():
    t0 = pd.Timestamp("2021-01-01 04:00:00")

    # No dropout timedeltas
    dropout_time = draw_dropout_time(t0, dropout_timedeltas=None, dropout_frac=1)
    assert dropout_time is None

    # Dropout fraction is 0
    dropout_timedeltas = [pd.Timedelta(-30, "min")]
    dropout_time = draw_dropout_time(t0, dropout_timedeltas=dropout_timedeltas, dropout_frac=0)
    assert dropout_time is None

    # No dropout timedeltas and dropout fraction is 0
    dropout_time = draw_dropout_time(t0, dropout_timedeltas=None, dropout_frac=0)
    assert dropout_time is None


@pytest.mark.parametrize("t0_str", ["12:00", "12:30", "13:00"])
def test_apply_dropout_time(da_sample, t0_str):
    dropout_time = pd.Timestamp(f"2024-01-01 {t0_str}")

    da_dropout = apply_dropout_time(da_sample, dropout_time)

    assert da_dropout.sel(time_utc=slice(None, dropout_time)).notnull().all()
    assert da_dropout.sel(time_utc=slice(dropout_time+pd.Timedelta(5, "min"), None)).isnull().all()
