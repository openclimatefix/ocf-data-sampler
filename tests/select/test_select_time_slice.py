import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp
from tests.conftest import NWP_FREQ


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_select_time_slice(da_sat_like, t0_str):
    """Test the basic functionality of select_time_slice"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    interval_start = pd.Timedelta(0, "min")
    interval_end = pd.Timedelta(60, "min")
    freq = pd.Timedelta("5min")

    # Expect to return these timestamps from the selection
    expected_datetimes = pd.date_range(t0 + interval_start, t0 + interval_end, freq=freq)

    # Make the selection
    sat_sample = select_time_slice(
        da_sat_like,
        t0=t0,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    # Check the returned times are as expected
    assert (sat_sample.time_utc == expected_datetimes).all()


@pytest.mark.parametrize("t0_str", ["00:00", "00:25", "11:00", "11:55"])
def test_select_time_slice_out_of_bounds(da_sat_like, t0_str):
    """Test the behaviour of select_time_slice when the selection is out of bounds"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    interval_start = pd.Timedelta(-30, "min")
    interval_end = pd.Timedelta(60, "min")
    freq = pd.Timedelta("5min")

    # The data is available between these times
    min_time = pd.Timestamp(da_sat_like.time_utc.min().item())
    max_time = pd.Timestamp(da_sat_like.time_utc.max().item())

    # Expect to return these timestamps within the requested range
    expected_datetimes = pd.date_range(
        max(t0 + interval_start, min_time),
        min(t0 + interval_end, max_time),
        freq=freq,
    )

    # Make the partially out of bounds selection
    sat_sample = select_time_slice(
        da_sat_like,
        t0=t0,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    # Check the returned times are as expected
    assert (sat_sample.time_utc == expected_datetimes).all()

    # Check all the values before the first timestamp available in the data are NaN
    all_nan_space = sat_sample.isnull().all(dim=("x_geostationary", "y_geostationary"))
    if expected_datetimes[0] < min_time:
        assert all_nan_space.sel(time_utc=slice(None, min_time - freq)).all(dim="time_utc")

    # Check all the values after the last timestamp available in the data are NaN
    if expected_datetimes[-1] > max_time:
        assert all_nan_space.sel(time_utc=slice(max_time + freq, None)).all(dim="time_utc")

    # Check that none of the values between the first and last available timestamp are NaN
    any_nan_space = sat_sample.isnull().any(dim=("x_geostationary", "y_geostationary"))
    assert not any_nan_space.sel(time_utc=slice(min_time, max_time)).any(dim="time_utc")


@pytest.mark.parametrize("t0_str", ["10:00", "11:00", "12:00"])
def test_select_time_slice_nwp_basic(da_nwp_like, t0_str):
    """Test the basic functionality of select_time_slice_nwp"""

    # Slice parameters
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    interval_start = pd.Timedelta(-6, "h")
    interval_end = pd.Timedelta(3, "h")
    freq = pd.Timedelta("1h")

    # Make the selection
    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        time_resolution=freq,
        interval_start=interval_start,
        interval_end=interval_end,
        dropout_timedeltas=None,
        dropout_frac=0,
    )

    # Check the target-times are as expected
    expected_target_times = pd.date_range(t0 + interval_start, t0 + interval_end, freq=freq)
    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-times are as expected
    # - Forecast frequency is `NWP_FREQ`, and we can't have selected future init-times
    expected_init_times = pd.to_datetime(
        [t if t < t0 else t0 for t in expected_target_times],
    ).floor(NWP_FREQ)
    assert (expected_init_times == da_slice.init_time_utc.values).all()


@pytest.mark.parametrize("dropout_hours", [1, 2, 5])
def test_select_time_slice_nwp_with_dropout(da_nwp_like, dropout_hours):
    """Test the functionality of select_time_slice_nwp with dropout"""

    t0 = pd.Timestamp("2024-01-02 12:00")
    interval_start = pd.Timedelta(-6, "h")
    interval_end = pd.Timedelta(3, "h")
    freq = pd.Timedelta("1h")
    dropout_timedelta = pd.Timedelta(f"-{dropout_hours}h")

    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        time_resolution=freq,
        interval_start=interval_start,
        interval_end=interval_end,
        dropout_timedeltas=[dropout_timedelta],
        dropout_frac=1,
    )

    # Check the target-times are as expected
    expected_target_times = pd.date_range(t0 + interval_start, t0 + interval_end, freq=freq)
    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-times are as expected considering the delay
    t0_delayed = t0 + dropout_timedelta
    expected_init_times = pd.to_datetime(
        [t if t < t0_delayed else t0_delayed for t in expected_target_times],
    ).floor(NWP_FREQ)
    assert (expected_init_times == da_slice.init_time_utc.values).all()


@pytest.fixture
def nwp_data_array_non_hourly():
    """NWP data array with non-hourly init times"""
    # Create a dummy NWP data array
    data = np.random.rand(2, 3, 2, 4, 5).astype(np.float32)
    init_time = pd.to_datetime(["2023-01-01 12:30", "2023-01-01 13:30"])
    step = pd.to_timedelta([0, 0.5, 1], unit="h")
    channel = ["t", "dswrf"]
    x = np.arange(4)
    y = np.arange(5)
    da = xr.DataArray(
        data,
        coords=[init_time, step, channel, x, y],
        dims=["init_time_utc", "step", "channel", "x_osgb", "y_osgb"],
    )
    return da


def test_select_time_slice_nwp_raises_error_for_early_target_time(nwp_data_array_non_hourly):
    """Test that select_time_slice_nwp raises a descriptive error for a too-early target time."""
    with pytest.raises(ValueError, match=r"Target time .* is before the first available init time"):
        select_time_slice_nwp(
            da=nwp_data_array_non_hourly,
            t0=pd.Timestamp("2023-01-01 13:00"),
            interval_start=pd.Timedelta("-1h"),
            interval_end=pd.Timedelta("0h"),
            time_resolution=pd.Timedelta("1h"),
        )


def test_select_time_slice_nwp_success(nwp_data_array_non_hourly):
    """Test that select_time_slice_nwp works for a valid t0"""
    da_sel = select_time_slice_nwp(
        da=nwp_data_array_non_hourly,
        t0=pd.Timestamp("2023-01-01 14:00"),
        interval_start=pd.Timedelta("-1h"),
        interval_end=pd.Timedelta("0h"),
        time_resolution=pd.Timedelta("1h"),
    )

    # Expected target times are 13:00 and 14:00.
    # For 13:00, init_time is 12:30 (step 30 mins).
    # For 14:00, init_time is 13:30 (step 30 mins).
    expected_steps = pd.TimedeltaIndex(["30min", "30min"])
    expected_init_times = pd.to_datetime(["2023-01-01 12:30", "2023-01-01 13:30"])
    assert da_sel.dims == ("time_utc", "channel", "x_osgb", "y_osgb")
    assert len(da_sel.step) == 2
    assert np.all(da_sel.step.values == expected_steps.values)
    assert np.all(da_sel.init_time_utc.values == expected_init_times.values)


def test_select_time_slice_nwp_no_available_init_times(nwp_data_array_non_hourly):
    """Test that select_time_slice_nwp raises error if no init times can cover the target period"""
    with pytest.raises(ValueError, match="Cannot get NWP data for target time"):
        select_time_slice_nwp(
            da=nwp_data_array_non_hourly,
            t0=pd.Timestamp(
                "2023-01-01 10:00",
            ),  # t0 is very early, no init_time can cover it
            interval_start=pd.Timedelta("0h"),
            interval_end=pd.Timedelta("1h"),
            time_resolution=pd.Timedelta("1h"),
        )


def test_select_time_slice_nwp_handles_extended_forecast_coverage(da_nwp_like):
    """
    Test that select_time_slice_nwp correctly selects an init_time that is much
    earlier than the requested time range, but whose forecast covers the range.
    """
    # Request a time range (14:00-15:00) that does not contain an init_time itself.
    # The latest init_time before this range is 12:00.
    t0 = pd.Timestamp("2024-01-02 13:00")
    start_interval = pd.Timedelta("1h")  # Start at 14:00
    end_interval = pd.Timedelta("2h")  # End at 15:00
    freq = pd.Timedelta("1h")

    # Make the selection
    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        time_resolution=freq,
        interval_start=start_interval,
        interval_end=end_interval,
    )

    # Check that the init_time for both target times (14:00, 15:00) is 12:00
    expected_init_time = pd.Timestamp("2024-01-02 12:00")
    assert (da_slice.init_time_utc == expected_init_time).all()

    # Check that the steps are correct (2h and 3h from 12:00)
    expected_steps = pd.to_timedelta(["2h", "3h"])
    assert (da_slice.step.values == expected_steps).all()

    # Verify the valid times are correct
    expected_valid_times = pd.to_datetime(["2024-01-02 14:00", "2024-01-02 15:00"])
    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times.values == expected_valid_times).all()
