import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.dropout import apply_sampled_dropout_time


@pytest.fixture(scope="module")
def da_sample():
    """Create dummy data which looks like satellite data"""

    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords={
            "time_utc": (["time_utc"], datetimes),
        },
    )
    return da_sat


def test_draw_dropout_time_none(da_sample):
    t0 = pd.Timestamp("2021-01-01 04:00:00")

    # Dropout fraction is 0
    dropout_timedeltas = [pd.Timedelta(-30, "min")]
    da_sample_dropout = apply_sampled_dropout_time(
        t0,
        dropout_timedeltas=dropout_timedeltas,
        dropout_frac=0,
        da=da_sample,
    )

    # Check data arrays are equal as dropout time would be None
    xr.testing.assert_equal(da_sample_dropout, da_sample)

    # No dropout timedeltas and dropout fraction is 0
    da_sample_dropout = apply_sampled_dropout_time(
        t0,
        dropout_timedeltas=[],
        dropout_frac=0,
        da=da_sample,
    )

    # Check data arrays are equal as dropout time would be None
    xr.testing.assert_equal(da_sample_dropout, da_sample)


@pytest.mark.parametrize("t0_str", ["12:30", "13:00", "13:30"])
def test_apply_sampled_dropout_time(da_sample, t0_str):
    t0_time = pd.Timestamp(f"2024-01-01 {t0_str}")
    dropout_time = t0_time + pd.Timedelta(minutes=-30)

    da_dropout = apply_sampled_dropout_time(
        t0_time,
        dropout_timedeltas=[pd.Timedelta(minutes=-30)],
        dropout_frac=1.0,
        da=da_sample,
    )

    assert da_dropout.sel(time_utc=slice(None, dropout_time)).notnull().all()
    assert (
        da_dropout.sel(time_utc=slice(dropout_time + pd.Timedelta(5, "min"), None)).isnull().all()
    )
