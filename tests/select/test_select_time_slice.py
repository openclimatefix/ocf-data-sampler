from ocf_data_sampler.select.select_time_slice import select_time_slice
from ocf_data_sampler.load.satellite import open_sat_data

from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import pytest


@pytest.fixture(scope="module")
def da_sat_like():
    # Create dummy data which looks like satellite data
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(x), len(y))),
        coords=dict(
            time_utc=(["time_utc"], datetimes),
            x_geostationary=(["x_geostationary"], x),
            y_geostationary=(["y_geostationary"], y),

        )
    )
    return da_sat


def test_select_time_slice(da_sat_like):
    t0 = pd.Timestamp("2024-01-02 12:00")

    forecast_duration = timedelta(minutes=0)
    history_duration = timedelta(minutes=30)
    freq = timedelta(minutes=5)

    # Expect to return these timestamps from the selection
    expected_datetimes = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)

    # Test history and forecast param usage
    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        sample_period_duration=freq,
    )

    assert (sat_sample.time_utc == expected_datetimes).all()

    # Test interval param usage
    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        interval_start=-history_duration,
        interval_end=forecast_duration,
        sample_period_duration=freq,
    )

    assert (sat_sample.time_utc == expected_datetimes).all()


def test_select_time_slice_out_of_bounds(da_sat_like):

    t0 = pd.Timestamp("2024-01-02 00:30")

    forecast_duration = timedelta(minutes=0)
    history_duration = timedelta(minutes=60)
    freq = timedelta(minutes=5)

    # Expect to return these timestamps from the selection
    expected_datetimes = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)

    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        sample_period_duration=freq,
        fill_selection=True
    )

    assert (sat_sample.time_utc == expected_datetimes).all()

    # Correct number of time steps are all NaN
    all_nan_space = sat_sample.isnull().all(dim=("x_geostationary", "y_geostationary"))

    # Check all the values before the first timestamp available in the data are NaN
    assert all_nan_space.sel(time_utc=slice(None, "2024-01-01 23:55")).all(dim="time_utc")

    # check all the values after the first timestamp available in the data are not NaN
    assert not all_nan_space.sel(time_utc=slice("2024-01-02 00:00", None)).any(dim="time_utc")
