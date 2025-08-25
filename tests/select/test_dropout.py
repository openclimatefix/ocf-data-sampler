import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.dropout import apply_history_dropout
from ocf_data_sampler.utils import minutes


@pytest.fixture(scope="module")
def da_sample():

    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords={"time_utc": (["time_utc"], datetimes)},
    )
    return da_sat


def test_apply_history_dropout_multiple_timedeltas(da_sample):
    t0 = da_sample.time_utc.values[-1]

    da_sample_dropout = apply_history_dropout(
        t0,
        dropout_timedeltas=minutes([-30, -45]),
        dropout_frac=1,
        da=da_sample,
    )

    latest_expected_cut_off = t0 + minutes(-30)

    assert (
        da_sample_dropout
        .sel(time_utc=slice(latest_expected_cut_off + minutes(5), None))
        .isnull()
        .all()
    )


def test_apply_history_dropout_none(da_sample):
    t0 = da_sample.time_utc.values[-1]

    da_sample_dropout = apply_history_dropout(
        t0,
        dropout_timedeltas=[minutes(-30)],
        dropout_frac=0,
        da=da_sample,
    )

    # Check data arrays are equal as dropout time would be None
    xr.testing.assert_equal(da_sample_dropout, da_sample)

    # No dropout timedeltas and dropout fraction is 0
    da_sample_dropout = apply_history_dropout(
        t0,
        dropout_timedeltas=[],
        dropout_frac=0,
        da=da_sample,
    )

    # Check data arrays are equal as dropout time would be None
    xr.testing.assert_equal(da_sample_dropout, da_sample)


def test_apply_history_dropout_list(da_sample):
    t0 = da_sample.time_utc.values[-1]

    da_sample_dropout = apply_history_dropout(
        t0,
        dropout_timedeltas=minutes([-30, -45]),
        dropout_frac=[0.5,0.5],
        da=da_sample,
    )

    latest_expected_cut_off = t0 + minutes(-30)

    assert (
        da_sample_dropout.sel(time_utc=slice(latest_expected_cut_off + minutes(5), None))
        .isnull()
        .all()
    )


@pytest.mark.parametrize("t0_str", ["12:30", "13:00", "13:30"])
def test_apply_history_dropout(da_sample, t0_str):
    t0_time = pd.Timestamp(f"2024-01-01 {t0_str}")
    dropout_time = t0_time + minutes(-30)

    da_dropout = apply_history_dropout(
        t0_time,
        dropout_timedeltas=[minutes(-30)],
        dropout_frac=1.0,
        da=da_sample,
    )

    assert da_dropout.sel(time_utc=slice(None, dropout_time)).notnull().all()
    assert (da_dropout.sel(time_utc=slice(dropout_time + minutes(5), t0_time)).isnull().all())
