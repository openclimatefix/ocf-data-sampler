import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import date_range, datetime_ceil, minutes
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


def test_select_time_slice_rejects_off_grid_request():
    """Test that off-grid requests are rejected instead of rounded"""

    t0 = np.datetime64("2024-01-01T12:30")
    interval_start = minutes(-120)
    interval_end = minutes(120)
    freq = minutes(60)

    # t0 is not in the time grid so this should raise an error
    times = date_range(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-02T00:00"),
        freq=freq,
    )
    da = make_dataarray(times)

    with pytest.raises(ValueError, match="Not all values"):
        _ = select_time_slice(
            da,
            t0=t0,
            interval_start=interval_start,
            interval_end=interval_end,
            time_resolution=freq,
        )


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
        dropout_timedeltas=None,
        dropout_frac=0,
    )

    # Check the target-times are as expected
    expected_target_times = datetime_ceil(
        date_range(t0 + interval_start, t0 + interval_end, freq=freq),
        freq=freq,
    )

    valid_times = da_slice.init_time_utc + da_slice.step
    assert (valid_times == expected_target_times).all()

    # Check the init-time is the first init time before the first target time
    init_times = da_nwp_like["init_time_utc"].values
    expected_init_time = init_times[init_times<=expected_target_times[0]][-1]
    assert (expected_init_time == da_slice["init_time_utc"].values)


@pytest.mark.parametrize("dropout_hours", [1, 2, 3, 5])
def test_select_time_slice_nwp_with_dropout(da_nwp_like, dropout_hours):
    """Test the functionality of select_time_slice_nwp with dropout"""

    t0 = np.datetime64("2024-01-02 12:00")
    interval_start = np.timedelta64(-2, "h")
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
    valid_times = da_slice["init_time_utc"] + da_slice["step"]
    assert (valid_times == expected_target_times).all()

    # Check the init-time is the first init time before the first target time whilst considering the
    # delay
    t0_delayed = min(t0 + dropout_timedelta, expected_target_times[0])
    init_times = da_nwp_like["init_time_utc"].values
    expected_init_time = init_times[init_times<=t0_delayed][-1]
    assert (expected_init_time == da_slice["init_time_utc"].values)


def test_select_time_slice_nwp_with_weighted_dropout_list(da_nwp_like):
    """List dropout probabilities should select the corresponding timedelta weights."""
    t0 = np.datetime64("2024-01-02 12:00")
    interval_start = np.timedelta64(-2, "h")
    interval_end = np.timedelta64(3, "h")
    freq = np.timedelta64(1, "h")

    da_slice = select_time_slice_nwp(
        da_nwp_like,
        t0,
        time_resolution=freq,
        interval_start=interval_start,
        interval_end=interval_end,
        dropout_timedeltas=[np.timedelta64(-1, "h"), np.timedelta64(-2, "h")],
        dropout_frac=[1.0, 0.0],
    )

    expected_target_times = date_range(t0 + interval_start, t0 + interval_end, freq=freq)
    valid_times = da_slice["init_time_utc"] + da_slice["step"]
    assert (valid_times == expected_target_times).all()

    t0_delayed = min(t0 + np.timedelta64(-1, "h"), expected_target_times[0])
    init_times = da_nwp_like["init_time_utc"].values
    expected_init_time = init_times[init_times <= t0_delayed][-1]
    assert (expected_init_time == da_slice["init_time_utc"].values)


def test_select_time_slice_nwp_rejects_invalid_weighted_dropout_inputs(da_nwp_like):
    """List dropout probabilities must satisfy sum and length constraints."""
    kwargs = {
        "da": da_nwp_like,
        "t0": np.datetime64("2024-01-02 12:00"),
        "time_resolution": np.timedelta64(1, "h"),
        "interval_start": np.timedelta64(-2, "h"),
        "interval_end": np.timedelta64(3, "h"),
    }

    with pytest.raises(ValueError, match="sum of `dropout_frac`"):
        select_time_slice_nwp(
            **kwargs,
            dropout_timedeltas=[np.timedelta64(-1, "h"), np.timedelta64(-2, "h")],
            dropout_frac=[0.8, 0.4],
        )

    with pytest.raises(ValueError, match="must have the same length"):
        select_time_slice_nwp(
            **kwargs,
            dropout_timedeltas=[np.timedelta64(-1, "h"), np.timedelta64(-2, "h")],
            dropout_frac=[0.5],
        )
