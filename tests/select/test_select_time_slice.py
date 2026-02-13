import numpy as np
import pytest

from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp
from ocf_data_sampler.time_utils import date_range, datetime_floor
from tests.conftest import NWP_FREQ


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_select_time_slice(da_sat_like, t0_str):
    """Test the basic functionality of select_time_slice"""

    # Slice parameters
    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = np.timedelta64(0, "m")
    interval_end = np.timedelta64(60, "m")
    freq = np.timedelta64(5, "m")

    # Expect to return these timestamps from the selection
    expected_datetimes = date_range(t0 + interval_start, t0 + interval_end, freq=freq)

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
    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = np.timedelta64(-30, "m")
    interval_end = np.timedelta64(60, "m")
    freq = np.timedelta64(5, "m")

    # The data is available between these times
    min_time = np.datetime64(da_sat_like.time_utc.values.min())
    max_time = np.datetime64(da_sat_like.time_utc.values.max())

    # Expect to return these timestamps within the requested range
    expected_datetimes = date_range(
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
    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = np.timedelta64(-6, "h")
    interval_end = np.timedelta64(3, "h")
    freq = np.timedelta64(1, "h")


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
    expected_target_times = date_range(t0 + interval_start, t0 + interval_end, freq=freq)
    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-times are as expected
    # - Forecast frequency is `NWP_FREQ`, and we can't have selected future init-times
    expected_init_times = datetime_floor(
        np.array([t if t < t0 else t0 for t in expected_target_times]),
        freq=NWP_FREQ,
    )
    assert (expected_init_times == da_slice.init_time_utc.values).all()


@pytest.mark.parametrize("dropout_hours", [1, 2, 5])
def test_select_time_slice_nwp_with_dropout(da_nwp_like, dropout_hours):
    """Test the functionality of select_time_slice_nwp with dropout"""

    t0 = np.datetime64("2024-01-02 12:00")
    interval_start = np.timedelta64(-6, "h")
    interval_end = np.timedelta64(3, "h")
    freq = np.timedelta64(1, "h")
    dropout_timedelta = np.timedelta64(-dropout_hours, "h")

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
    expected_target_times = date_range(t0 + interval_start, t0 + interval_end, freq=freq)
    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-times are as expected considering the delay
    t0_delayed = t0 + dropout_timedelta
    expected_init_times = datetime_floor(
        np.array([t if t < t0_delayed else t0_delayed for t in expected_target_times]),
        freq=NWP_FREQ,
    )
    assert (expected_init_times == da_slice.init_time_utc.values).all()
