import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.dropout import simulate_dropout


@pytest.fixture(scope="module")
def da_sample():
    """Create dummy data which looks like satellite data"""
    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")
    da = xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords={"time_utc": datetimes}
    )
    return da


@pytest.fixture
def t0():
    return pd.Timestamp("2021-01-01 04:00:00")


def test_simulate_dropout_draw_time(t0):
    dropout_timedeltas = pd.to_timedelta([-30, -60], unit="min")
    times = pd.date_range(t0 - pd.Timedelta("1h"), t0 + pd.Timedelta("30min"), freq="10min")
    ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})

    ds_dropout = simulate_dropout(ds, t0, dropout_timedeltas, dropout_frac=1)
    non_nan = ds_dropout.where(~xr.ufuncs.isnan(ds_dropout), drop=True)
    dropout_time = non_nan.time_utc.values[-1]
    offset = pd.Timestamp(dropout_time) - t0
    assert offset in dropout_timedeltas


def test_simulate_dropout_draw_time_none(t0):
    times = pd.date_range(t0 - pd.Timedelta("30min"), t0 + pd.Timedelta("30min"), freq="10min")
    ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})

    # Test dropout_frac=0
    result = simulate_dropout(ds, t0, [pd.Timedelta("-30min")], dropout_frac=0)
    xr.testing.assert_equal(result, ds)

    # Test empty dropout_timedeltas
    result = simulate_dropout(ds, t0, [], dropout_frac=0)
    xr.testing.assert_equal(result, ds)


@pytest.mark.parametrize(
    "delta,expect_dropout",
    [
        (None, False),            # No dropout case
        (pd.Timedelta(0), True),  # Dropout exactly at t0
        (pd.Timedelta("-5min"), True),
        (pd.Timedelta("-10min"), True),
    ]
)
def test_simulate_dropout_apply(t0, delta, expect_dropout):
    """Test both dropout and no-dropout scenarios."""
    # Create test data
    times = pd.date_range(t0 - pd.Timedelta("10min"), t0 + pd.Timedelta("10min"), freq="5min")
    ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})

    if delta is None:
        # Test no dropout configuration
        ds_dropout = simulate_dropout(ds, t0, [], dropout_frac=0)
    else:
        # Test with forced dropout
        ds_dropout = simulate_dropout(ds, t0, [delta], dropout_frac=1)
        dropout_time = t0 + delta

    if expect_dropout:
        # Verify post-dropout NaNs
        after_time = dropout_time + pd.Timedelta("5min")
        assert ds_dropout.sel(time_utc=slice(after_time, None)).isnull().all()
        # Verify pre-dropout data intact
        assert ds_dropout.sel(time_utc=slice(None, dropout_time)).notnull().all()
    else:
        # Verify no modifications
        assert ds_dropout.identical(ds)


def test_simulate_dropout_invalid_positive_delta(t0):
    """Test validation rejects positive timedeltas."""
    with pytest.raises(ValueError, match="Dropout timedeltas must be negative"):
        simulate_dropout(xr.DataArray([1]), t0, [pd.Timedelta("30min")], dropout_frac=1)