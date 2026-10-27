import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import date_range, datetime_floor, minutes
from ocf_data_sampler.select.time_slice import select_time_slice, select_time_slice_nwp


def make_dataarray(times):
    return xr.DataArray(np.zeros(len(times)), coords={"time_utc": times}, dims="time_utc")


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_select_time_slice(da_sat_like, t0_str):
    """Test the basic functionality of select_time_slice"""

    # Slice parameters
    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = minutes(0)
    interval_end = minutes(60)
    freq = minutes(5)

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


@pytest.mark.parametrize("t0_str", ["00:00", "00:25", "23:05", "23:55"])
def test_select_time_slice_rejects_out_of_bounds(da_sat_like, t0_str):
    """Test the behaviour of select_time_slice when the selection is out of bounds"""

    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = minutes(-30)
    interval_end = minutes(60)
    freq = minutes(5)

    with pytest.raises(ValueError, match=r"Not all values in .* exist in array .*"):
        # Make the partially out of bounds selection
        _ = select_time_slice(
            da_sat_like,
            t0=t0,
            interval_start=interval_start,
            interval_end=interval_end,
            time_resolution=freq,
        )


@pytest.mark.parametrize(
    "offsets",
    [[0, 45, 60], [0, 60]],
    ids=["wrong-interior-timestamp", "missing-interior-timestamp"],
)
def test_select_time_slice_rejects_incorrect_timestamps(offsets):
    """Test that incorrect interior timestamps are rejected"""

    t0 = np.datetime64("2024-01-01T00:00")
    interval_start = minutes(0)
    interval_end = minutes(60)
    freq = minutes(30)

    da = make_dataarray(t0 + minutes(offsets))

    with pytest.raises(ValueError, match="do not match time steps"):
        # Make the selection with incorrect timestamps
        _ = select_time_slice(
            da,
            t0=t0,
            interval_start=interval_start,
            interval_end=interval_end,
            time_resolution=freq,
        )


def test_select_time_slice_floors_off_grid_request():
    """Test that off-grid t0 is floored before selecting the inclusive window"""

    t0 = np.datetime64("2024-01-01T12:30")
    interval_start = minutes(-120)
    interval_end = minutes(120)
    freq = minutes(60)

    times = date_range(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-02T00:00"),
        freq=freq,
    )
    da = make_dataarray(times)

    da_slice = select_time_slice(
        da,
        t0=t0,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    expected_datetimes = date_range(
        np.datetime64("2024-01-01T10:00"), np.datetime64("2024-01-01T14:00"), freq=freq
    )
    np.testing.assert_array_equal(da_slice.time_utc.values, expected_datetimes)


@pytest.mark.parametrize("t0_str", ["10:00", "10:30", "11:00", "11:15", "12:00"])
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
        delay=None,
    )

    # Check the target-times are as expected
    expected_target_times = datetime_floor(
        date_range(t0 + interval_start, t0 + interval_end, freq=freq),
        freq=freq,
    )

    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-time is the first init time before the first target time
    init_times = da_nwp_like["init_time_utc"].values
    expected_init_time = init_times[init_times<=expected_target_times[0]][-1]
    assert (expected_init_time == da_slice["init_time_utc"].values)


@pytest.mark.parametrize(
    ("t0_str", "delay_minutes", "interval_start_minutes", "expected_init_time_str"),
    [
        ("12:00", 60, -120, "09:00"),
        ("12:00", 120, -120, "09:00"),
        ("12:00", 180, -120, "09:00"),
        ("12:00", 300, -120, "06:00"),
        ("12:30", 30, 0, "12:00"),  # At 12:30, the 12:00 forecast will be available.
    ],
)
def test_select_time_slice_nwp_with_delay(
    da_nwp_like, t0_str, delay_minutes, interval_start_minutes, expected_init_time_str
):
    """Test the functionality of select_time_slice_nwp with delayed availability"""

    t0 = np.datetime64(f"2024-01-02 {t0_str}")
    interval_start = minutes(interval_start_minutes)
    interval_end = np.timedelta64(3, "h")
    freq = np.timedelta64(1, "h")
    delay_timedelta = minutes(-delay_minutes)

    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        time_resolution=freq,
        interval_start=interval_start,
        interval_end=interval_end,
        delay=-delay_timedelta,
    )

    # Check the target-times are as expected
    expected_target_times = datetime_floor(
        date_range(t0 + interval_start, t0 + interval_end, freq=freq),
        freq=freq,
    )
    valid_times = da_slice["init_time_utc"] + da_slice["step"]
    assert (valid_times == expected_target_times).all()

    expected_init_time = np.datetime64(f"2024-01-02 {expected_init_time_str}")
    assert (expected_init_time == da_slice["init_time_utc"].values)
