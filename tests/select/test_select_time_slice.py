from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp

import numpy as np
import pandas as pd
import xarray as xr
import pytest


NWP_FREQ = pd.Timedelta("3H")

@pytest.fixture(scope="module")
def da_sat_like():
    """Create dummy data which looks like satellite data"""
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


@pytest.fixture(scope="module")
def da_nwp_like():
    """Create dummy data which looks like NWP data"""

    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq=NWP_FREQ)
    steps = pd.timedelta_range("0H", "16H", freq="1H")
    channels = ["t", "dswrf"]

    da_nwp = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(steps), len(channels), len(x), len(y))),
        coords=dict(
            init_time_utc=(["init_time_utc"], datetimes),
            step=(["step"], steps),
            channel=(["channel"], channels),
            x_osgb=(["x_osgb"], x),
            y_osgb=(["y_osgb"], y),
        )
    )
    return da_nwp


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_select_time_slice(da_sat_like, t0_str):
    """Test the basic functionality of select_time_slice"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    forecast_duration = pd.Timedelta("0min")
    history_duration = pd.Timedelta("60min")
    freq = pd.Timedelta("5min")

    # Expect to return these timestamps from the selection
    expected_datetimes = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)

    # Make the selection using the `[x]_duration` parameters
    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        sample_period_duration=freq,
    )

    # Check the returned times are as expected
    assert (sat_sample.time_utc == expected_datetimes).all()

    # Make the selection using the `interval_[x]` parameters
    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        interval_start=-history_duration,
        interval_end=forecast_duration,
        sample_period_duration=freq,
    )

    # Check the returned times are as expected
    assert (sat_sample.time_utc == expected_datetimes).all()


@pytest.mark.parametrize("t0_str", ["00:00", "00:25", "11:00", "11:55"])
def test_select_time_slice_out_of_bounds(da_sat_like, t0_str):
    """Test the behaviour of select_time_slice when the selection is out of bounds"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    forecast_duration = pd.Timedelta("30min")
    history_duration = pd.Timedelta("60min")
    freq = pd.Timedelta("5min")

    # The data is available between these times
    min_time = da_sat_like.time_utc.min()
    max_time = da_sat_like.time_utc.max()

    # Expect to return these timestamps from the selection
    expected_datetimes = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)

    # Make the partially out of bounds selection
    sat_sample = select_time_slice(
        ds=da_sat_like,
        t0=t0,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        sample_period_duration=freq,
        fill_selection=True
    )

    # Check the returned times are as expected
    assert (sat_sample.time_utc == expected_datetimes).all()

    
    # Check all the values before the first timestamp available in the data are NaN
    all_nan_space = sat_sample.isnull().all(dim=("x_geostationary", "y_geostationary"))
    if expected_datetimes[0] < min_time:    
        assert all_nan_space.sel(time_utc=slice(None, min_time-freq)).all(dim="time_utc") 

    # Check all the values before the first timestamp available in the data are NaN
    if expected_datetimes[-1] > max_time:    
        assert all_nan_space.sel(time_utc=slice(max_time+freq, None)).all(dim="time_utc")   

    # Check that none of the values between the first and last available timestamp are NaN
    any_nan_space = sat_sample.isnull().any(dim=("x_geostationary", "y_geostationary"))
    assert not any_nan_space.sel(time_utc=slice(min_time, max_time)).any(dim="time_utc")


@pytest.mark.parametrize("t0_str", ["10:00", "11:00", "12:00"])
def test_select_time_slice_nwp_basic(da_nwp_like, t0_str):
    """Test the basic functionality of select_time_slice_nwp"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    forecast_duration = pd.Timedelta("6H")
    history_duration = pd.Timedelta("3H")
    freq = pd.Timedelta("1H")

    # Make the selection
    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        sample_period_duration=freq,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        dropout_timedeltas = None,
        dropout_frac = 0,
        accum_channels = [],
        channel_dim_name = "channel",
    )

    # Check the target-times are as expected
    expected_target_times = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)
    assert (da_slice.target_time_utc==expected_target_times).all()

    # Check the init-times are as expected
    # - Forecast frequency is `NWP_FREQ`, and we can't have selected future init-times
    expected_init_times = pd.to_datetime(
        [t if t<t0 else t0 for t in expected_target_times]
    ).floor(NWP_FREQ)
    assert (da_slice.init_time_utc==expected_init_times).all()


@pytest.mark.parametrize("dropout_hours", [1, 2, 5])
def test_select_time_slice_nwp_with_dropout(da_nwp_like, dropout_hours):
    """Test the functionality of select_time_slice_nwp with dropout"""

    t0 = pd.Timestamp("2024-01-02 12:00")
    forecast_duration = pd.Timedelta("6H")
    history_duration = pd.Timedelta("3H")
    freq = pd.Timedelta("1H")
    dropout_timedelta = pd.Timedelta(f"-{dropout_hours}H")

    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        sample_period_duration=freq,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        dropout_timedeltas = [dropout_timedelta],
        dropout_frac = 1,
        accum_channels = [],
        channel_dim_name = "channel",
    )

    # Check the target-times are as expected
    expected_target_times = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)
    assert (da_slice.target_time_utc==expected_target_times).all()

    # Check the init-times are as expected considering the delay
    t0_delayed = t0 + dropout_timedelta
    expected_init_times = pd.to_datetime(
        [t if t<t0_delayed else t0_delayed for t in expected_target_times]
    ).floor(NWP_FREQ)
    assert (da_slice.init_time_utc==expected_init_times).all()


@pytest.mark.parametrize("t0_str", ["10:00", "11:00", "12:00"])
def test_select_time_slice_nwp_with_dropout_and_accum(da_nwp_like, t0_str):
    """Test the functionality of select_time_slice_nwp with dropout and accumulated variables"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    forecast_duration = pd.Timedelta("6H")
    history_duration = pd.Timedelta("3H")
    freq = pd.Timedelta("1H")
    dropout_timedelta = pd.Timedelta("-2H")

    t0_delayed = (t0 + dropout_timedelta).floor(NWP_FREQ)

    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        sample_period_duration=freq,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        dropout_timedeltas=[dropout_timedelta],
        dropout_frac=1,
        accum_channels=["dswrf"],
        channel_dim_name="channel",
    )

    # Check the target-times are as expected
    expected_target_times = pd.date_range(t0 - history_duration, t0 + forecast_duration, freq=freq)
    assert (da_slice.target_time_utc==expected_target_times).all()

    # Check the init-times are as expected considering the delay
    expected_init_times = pd.to_datetime(
        [t if t<t0_delayed else t0_delayed for t in expected_target_times]
    ).floor(NWP_FREQ)
    assert (da_slice.init_time_utc==expected_init_times).all()

    # Check channels are as expected
    assert (da_slice.channel.values == ["t", "diff_dswrf"]).all()

    # Check the accummulated channel has been differenced correctly

    # This part of the data is pulled from the init-time: t0_delayed
    da_slice_accum = da_slice.sel(
        target_time_utc=slice(t0_delayed, None), 
        channel="diff_dswrf"
    )

    # Get the original data for the t0_delayed init-time, and diff it along steps
    # then select the steps which are expected to be used in the above slice
    da_orig_diffed = (
        da_nwp_like.sel(
            init_time_utc=t0_delayed, 
            channel="dswrf", 
        ).diff(dim="step", label="lower")
        .sel(step=slice(t0-t0_delayed - history_duration, t0-t0_delayed + forecast_duration))
    )

    # Check the values are the same
    assert (da_slice_accum.values == da_orig_diffed.values).all()

    # Check the non-accummulated channel has not been differenced

    # This part of the data is pulled from the init-time: t0_delayed
    da_slice_nonaccum = da_slice.sel(
        target_time_utc=slice(t0_delayed, None), 
        channel="t"
    )

    # Get the original data for the t0_delayed init-time, and select the steps which are expected 
    # to be used in the above slice
    da_orig = (
        da_nwp_like.sel(
            init_time_utc=t0_delayed, 
            channel="t", 
        )
        .sel(step=slice(t0-t0_delayed - history_duration, t0-t0_delayed + forecast_duration))
    )

    # Check the values are the same
    assert (da_slice_nonaccum.values == da_orig.values).all()


