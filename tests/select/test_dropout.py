import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.dropout import simulate_dropout

@pytest.fixture(scope="module")
def da_sample():
    """Create dummy data which looks like satellite data"""

    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")

    da = xr.DataArray(np.random.normal(size=(len(datetimes),)), coords={"time_utc": datetimes})
    return da

def infer_dropout_time(ds, t0):
    """
    Helper function to infer the effective dropout time from a dataset by finding the latest
    time for which the data is not NaN.
    """
    non_nan = ds.where(~xr.ufuncs.isnan(ds), drop=True)
    if non_nan.size > 0:
        return pd.Timestamp(non_nan.time_utc.values[-1])
    return t0

def test_simulate_dropout_draw_time():
    """
    Test that when dropout_frac is 1, the effective dropout time is t0 plus one of the
    provided negative timedeltas.
    """
    t0 = pd.Timestamp("2021-01-01 04:00:00")
    
    dropout_timedeltas = pd.to_timedelta([-30, -60], unit="min")
    # Create a dataset spanning from t0 - 1h to t0 + 30min.
    times = pd.date_range(t0 - pd.Timedelta("1h"), t0 + pd.Timedelta("30min"), freq="10min")
    ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})

    ds_dropout = simulate_dropout(ds, t0, dropout_timedeltas, dropout_frac=1)
    dropout_time = infer_dropout_time(ds_dropout, t0)
    offset = dropout_time - t0
    assert offset in dropout_timedeltas

def test_simulate_dropout_draw_time_none():
    """
    Test that when dropout_frac is 0 the dropout time remains t0, meaning that only data
    with time_utc <= t0 is preserved.
    """
    t0 = pd.Timestamp("2021-01-01 04:00:00")
    dropout_timedeltas = [pd.Timedelta(-30, "min")]
    times = pd.date_range("2021-01-01 03:00", periods=10, freq="10min")
    ds = xr.DataArray(np.arange(10), coords={"time_utc": times})

    ds_dropout = simulate_dropout(ds, t0, dropout_timedeltas, dropout_frac=0)
    ds_expected = ds.where(ds.time_utc <= t0)
    xr.testing.assert_equal(ds_dropout, ds_expected)

def test_simulate_dropout_partial():
    """
    Test that with dropout_frac between 0 and 1, both outcomes (dropout applied and not applied)
    occur.
    """
    t0 = pd.Timestamp("2021-01-01 04:00:00")
    dropout_timedeltas = pd.to_timedelta([-30, -60], unit="min")
    outcomes = set()

    for _ in range(100):
        times = pd.date_range(t0 - pd.Timedelta("1h"), t0 + pd.Timedelta("30min"), freq="10min")
        ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})
        ds_dropout = simulate_dropout(ds, t0, dropout_timedeltas, dropout_frac=2/3)
        dropout_time = infer_dropout_time(ds_dropout, t0)
        outcomes.add(dropout_time - t0)

    # Ensure that at least one dropout outcome is different from t0.
    assert any(delta != pd.Timedelta(0) for delta in outcomes)

def test_simulate_dropout_apply():
    """
    Test that the dropout is applied correctly by forcing a specific dropout time.
    We do this by providing a single negative timedelta.
    """
    t0 = pd.Timestamp("2021-01-01 04:00:00")
    # Use parameterized negative offsets: zero (no dropout), -5min, -10min.
    for delta in [pd.Timedelta(0), pd.Timedelta(-5, "min"), pd.Timedelta(-10, "min")]:
        times = pd.date_range("2021-01-01 03:50", "2021-01-01 04:10", freq="5min")
        ds = xr.DataArray(np.arange(len(times)), coords={"time_utc": times})
        ds_dropout = simulate_dropout(ds, t0, [delta], dropout_frac=1)
        dropout_time = t0 + delta
        # Data points at or before the dropout time should be non-NaN.
        assert ds_dropout.sel(time_utc=slice(None, dropout_time)).notnull().all()
        # Data points after the dropout time (by at least one step) should be NaN.
        after_time = dropout_time + pd.Timedelta(5, "min")
        assert ds_dropout.sel(time_utc=slice(after_time, None)).isnull().all()

def test_simulate_dropout_invalid_positive_delta():
    """
    Test that a ValueError is raised if any dropout_timedelta is positive.
    """
    t0 = pd.Timestamp("2021-01-01 04:00:00")
    with pytest.raises(ValueError, match="Dropout timedeltas must be negative"):
        simulate_dropout(xr.DataArray([1]), t0, [pd.Timedelta("30min")], dropout_frac=1)
