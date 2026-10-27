import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import date_range, minutes
from ocf_data_sampler.select.time_periods import (
    fill_time_periods,
    find_contiguous_t0_periods,
    find_contiguous_t0_periods_nwp,
    intersect_time_periods,
)
from ocf_data_sampler.select.time_slice import select_time_slice, select_time_slice_nwp
from tests.conftest import to_datetime64


def construct_time_periods_df(start_dt: list[str], end_dt: list[str]) -> pd.DataFrame:
    """Helper function to construct a DataFrame of time periods

    Args:
        start_dt: List of period start datetimes
        end_dt: List of period end datetimes

    Returns:
        pd.DataFrame: DataFrame with start and end datetimes columns where each period is a row
    """
    return pd.DataFrame(
        {
            "start_dt": to_datetime64(start_dt),
            "end_dt": to_datetime64(end_dt),
        },
    )


def test_find_contiguous_t0_periods():

    # Typical case with some missing time stamps in the middle of the range
    freq = minutes(5)
    interval_start = minutes(-60)
    interval_end = minutes(15)

    all_datetimes = date_range(
        to_datetime64("2023-01-01 12:00"),
        to_datetime64("2023-01-01 17:00"),
        freq=freq,
    )
    missing_datetimes = to_datetime64(["2023-01-01 12:25", "2023-01-01 12:30", "2023-01-01 14:30"])
    datetimes = np.setdiff1d(all_datetimes, missing_datetimes)

    periods = find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    expected_results = construct_time_periods_df(
        start_dt=["2023-01-01 13:35", "2023-01-01 15:35"],
        end_dt=["2023-01-01 14:15", "2023-01-01 16:50"],
    )

    assert periods.equals(expected_results)

    # A window longer than the available history cannot produce a sample.
    assert find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=np.timedelta64(-24, "h"),
        interval_end=np.timedelta64(24, "h"),
        time_resolution=freq,
    ).empty
    assert find_contiguous_t0_periods(
        datetimes=datetimes[:0],
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    ).empty

    # This is a stand in for where we just need a single satellite image from 5 minutes ago
    interval_start = minutes(-5)
    interval_end = minutes(-5)

    datetimes = to_datetime64(
        [
            "2023-01-01 12:00",
            "2023-01-01 12:05",
            "2023-01-01 12:10",
            "2023-01-01 12:20",
        ],
    )

    periods = find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    expected_results = construct_time_periods_df(
        start_dt=["2023-01-01 12:05", "2023-01-01 12:25"],
        end_dt=["2023-01-01 12:20", "2023-01-01 12:30"],
    )

    assert periods.equals(expected_results)



def test_find_contiguous_t0_periods_keeps_exact_length_period():
    freq = minutes(5)
    interval_start = minutes(0)
    interval_end = minutes(20)

    datetimes = date_range(
        to_datetime64("2023-01-01 00:00"),
        to_datetime64("2023-01-01 00:20"),
        freq=freq,
    )

    periods = find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=freq,
    )

    expected_results = construct_time_periods_df(
        start_dt=["2023-01-01 00:00"],
        end_dt=["2023-01-01 00:05"],
    )

    assert periods.equals(expected_results)


def test_find_contiguous_t0_periods_nwp():
    # These are the expected results of the test
    exp_res1 = construct_time_periods_df(
        start_dt=["2023-01-01 03:00", "2023-01-02 03:00"],
        end_dt=["2023-01-01 21:00", "2023-01-03 06:00"],
    )
    exp_res2 = construct_time_periods_df(
        start_dt=["2023-01-01 05:00", "2023-01-02 05:00", "2023-01-02 14:00"],
        end_dt=["2023-01-01 21:00", "2023-01-02 12:00", "2023-01-03 06:00"],
    )
    exp_res3 = construct_time_periods_df(
        start_dt=["2023-01-01 05:00", "2023-01-01 11:00", "2023-01-02 05:00", "2023-01-02 14:00"],
        end_dt=["2023-01-01 09:00", "2023-01-01 18:00", "2023-01-02 09:00", "2023-01-03 03:00"],
    )
    exp_res4 = construct_time_periods_df(
        start_dt=[
            "2023-01-01 05:00", "2023-01-01 11:00", "2023-01-01 14:00", "2023-01-02 05:00",
            "2023-01-02 14:00", "2023-01-02 17:00", "2023-01-02 20:00", "2023-01-02 23:00",
        ],
        end_dt=[
            "2023-01-01 06:00", "2023-01-01 12:00", "2023-01-01 15:00", "2023-01-02 06:00",
            "2023-01-02 15:00", "2023-01-02 18:00", "2023-01-02 21:00", "2023-01-03 00:00",
        ],
    )
    exp_res5 = construct_time_periods_df(
        start_dt=["2023-01-01 06:00", "2023-01-01 12:00", "2023-01-02 06:00", "2023-01-02 15:00"],
        end_dt=["2023-01-01 09:00", "2023-01-01 18:00", "2023-01-02 09:00", "2023-01-03 03:00"],
    )

    expected_results = [exp_res1, exp_res2, exp_res3, exp_res4, exp_res5]

    # Create 3-hourly init times with a few time stamps missing
    init_times = to_datetime64([
        "2023-01-01 03:00",
        "2023-01-01 09:00",
        "2023-01-01 12:00",
        "2023-01-02 03:00",
        "2023-01-02 12:00",
        "2023-01-02 15:00",
        "2023-01-02 18:00",
        "2023-01-02 21:00",
    ])

    first_forecast_step = np.timedelta64(0, "h")
    last_forecast_step = np.timedelta64(36, "h")

    interval_end = np.timedelta64(3, "h")

    # Choose some history durations and staleness limits
    history_durations_hr = [0, 2, 2, 2, 2]
    staleness_limits_hr = [9, 9, 6, 3, 6]
    max_delays_hr = [0, 0, 0, 0, 3]

    for i, expected in enumerate(expected_results):
        interval_start = np.timedelta64(-history_durations_hr[i], "h")
        staleness_limit = np.timedelta64(staleness_limits_hr[i], "h")
        max_delay = np.timedelta64(max_delays_hr[i], "h")

        time_periods = find_contiguous_t0_periods_nwp(
            init_times=init_times,
            interval_start=interval_start,
            interval_end=interval_end,
            first_forecast_step=first_forecast_step,
            last_forecast_step=last_forecast_step,
            time_resolution=np.timedelta64(1, "h"),
            staleness_limit=staleness_limit,
            max_delay=max_delay,
        )

        # Check if results are as expected
        assert time_periods.equals(expected)


def test_find_contiguous_t0_periods_nwp_forecast_too_short():
    """Test that a forecast too short to serve any t0 raises rather than emitting empty periods."""
    init_times = date_range(
        to_datetime64("2023-01-01 00:00"),
        to_datetime64("2023-01-02 00:00"),
        freq=np.timedelta64(6, "h")
    )

    # The forecast only reaches 2 hours ahead, but each sample needs 3 hours of future data
    with pytest.raises(ValueError, match="no t0s are available"):
        find_contiguous_t0_periods_nwp(
            init_times=init_times,
            interval_start=np.timedelta64(0, "h"),
            interval_end=np.timedelta64(3, "h"),
            first_forecast_step=np.timedelta64(0, "h"),
            last_forecast_step=np.timedelta64(2, "h"),
            time_resolution=np.timedelta64(1, "h"),
        )

    # The same failure via a staleness_limit shorter than the wait imposed by first_forecast_step
    with pytest.raises(ValueError, match="no t0s are available"):
        find_contiguous_t0_periods_nwp(
            init_times=init_times,
            interval_start=np.timedelta64(0, "h"),
            interval_end=np.timedelta64(3, "h"),
            first_forecast_step=np.timedelta64(6, "h"),
            last_forecast_step=np.timedelta64(36, "h"),
            time_resolution=np.timedelta64(1, "h"),
            staleness_limit=np.timedelta64(3, "h"),
        )


def test_intersect_time_periods_with_2_inputs():
    def assert_expected_result_with_reverse(a, b, expected_result):
        """Assert the calculated intersection is as expected with and without a and b switched"""
        assert intersect_time_periods([a, b]).equals(expected_result)
        assert intersect_time_periods([b, a]).equals(expected_result)

    # a: [----)
    # b:  [-)
    a = construct_time_periods_df(start_dt=["2025-01-01 00:00"], end_dt=["2025-01-01 12:00"])
    b = construct_time_periods_df(start_dt=["2025-01-01 03:00"], end_dt=["2025-01-01 06:00"])
    assert_expected_result_with_reverse(a, b, expected_result=b)

    # a:   [----)
    # b: [--)
    a = construct_time_periods_df(start_dt=["2025-01-01 12:00"], end_dt=["2025-01-01 18:00"])
    b = construct_time_periods_df(start_dt=["2025-01-01 03:00"], end_dt=["2025-01-01 15:00"])
    exp_res = construct_time_periods_df(start_dt=["2025-01-01 12:00"], end_dt=["2025-01-01 15:00"])
    assert_expected_result_with_reverse(a, b, expected_result=exp_res)

    # a:      [--)
    # b:   [--)
    a = construct_time_periods_df(start_dt=["2025-01-01 12:00"], end_dt=["2025-01-01 18:00"])
    b = construct_time_periods_df(start_dt=["2025-01-01 00:00"], end_dt=["2025-01-01 12:00"])
    exp_res = construct_time_periods_df([], [])
    assert_expected_result_with_reverse(a, b, expected_result=exp_res)

    # a:       [-)
    # b:   [-)
    a = construct_time_periods_df(start_dt=["2025-01-01 00:00"], end_dt=["2025-01-01 00:30"])
    b = construct_time_periods_df(start_dt=["2025-01-01 06:00"], end_dt=["2025-01-01 06:30"])
    exp_res = construct_time_periods_df([], [])  # no intersection
    assert_expected_result_with_reverse(a, b, expected_result=exp_res)

    # An empty operand makes the intersection empty in either order.
    b = construct_time_periods_df([], [])
    assert_expected_result_with_reverse(a, b, expected_result=b)


def test_intersect_time_periods_with_many_inputs():
    periods_1 = construct_time_periods_df(
        start_dt=["2023-01-01 05:00", "2023-01-01 14:10"],
        end_dt=["2023-01-01 13:35", "2023-01-01 18:00"],
    )

    periods_2 = construct_time_periods_df(
        start_dt=["2023-01-01 12:00"],
        end_dt=["2023-01-02 00:00"],
    )

    periods_3 = construct_time_periods_df(
        start_dt=["2023-01-01 00:00", "2023-01-01 13:00"],
        end_dt=["2023-01-01 12:30", "2023-01-01 23:00"],
    )

    expected_result = construct_time_periods_df(
        start_dt=["2023-01-01 12:00", "2023-01-01 13:00", "2023-01-01 14:10"],
        end_dt=["2023-01-01 12:30", "2023-01-01 13:35", "2023-01-01 18:00"],
    )

    result = intersect_time_periods([periods_1, periods_2, periods_3])

    # Check if results are as expected
    assert result.equals(expected_result)


def test_fill_time_periods():
    time_periods = pd.DataFrame(
        {
            "start_dt": to_datetime64([
                "2021-01-01 04:10:00",
                "2021-01-01 09:00:00",
                "2021-01-01 09:15:00",
                "2021-01-01 12:00:00",
            ]),
            "end_dt": to_datetime64([
                "2021-01-01 06:00:00",
                "2021-01-01 09:00:00",
                "2021-01-01 09:20:00",
                "2021-01-01 14:45:00",
            ]),
        },
    )

    filled = fill_time_periods(time_periods, freq=minutes(30))

    expected = to_datetime64(
        [
            "2021-01-01 04:30",
            "2021-01-01 05:00",
            "2021-01-01 05:30",
            "2021-01-01 12:00",
            "2021-01-01 12:30",
            "2021-01-01 13:00",
            "2021-01-01 13:30",
            "2021-01-01 14:00",
            "2021-01-01 14:30",
        ],
    )

    assert np.array_equal(filled, expected)

    filled = fill_time_periods(construct_time_periods_df([], []), freq=minutes(30))
    assert len(filled) == 0


def test_find_contiguous_t0_periods_matches_slicing():
    """Check that calculated t0 availability agrees with slicing a timeseries with a gap."""
    interval_start = minutes(-30)
    interval_end = minutes(60)
    time_resolution = minutes(30)
    t0_freq = minutes(30)

    all_datetimes = date_range(
        to_datetime64("2023-01-02T00:00"),
        to_datetime64("2023-01-02T05:00"),
        freq=time_resolution,
    )
    # Removing 02:00 leaves exactly one full window before the gap.
    datetimes = np.setdiff1d(all_datetimes, to_datetime64(["2023-01-02T02:00"]))
    da = xr.DataArray(np.zeros(len(datetimes)), dims="time_utc", coords={"time_utc": datetimes})

    # Find the available t0 times for the data
    periods = find_contiguous_t0_periods(
        datetimes=datetimes,
        interval_start=interval_start,
        interval_end=interval_end,
        time_resolution=time_resolution,
    )
    calculated_t0_times = fill_time_periods(periods, freq=t0_freq)

    # Test slicing out a range of potential t0 times to see if the this agrees with the t0 times
    # calculated above
    potential_t0s = date_range(
        to_datetime64("2023-01-01T23:00"),
        to_datetime64("2023-01-02T06:00"),
        freq=t0_freq
    )
    sliceable_t0s = []
    for t0 in potential_t0s:
        try:
            _ = select_time_slice(
                da,
                t0,
                interval_start=interval_start,
                interval_end=interval_end,
                time_resolution=time_resolution,
            )
        except ValueError:
            continue

        sliceable_t0s.append(t0)

    np.testing.assert_array_equal(sliceable_t0s, calculated_t0_times)


def test_find_contiguous_t0_periods_nwp_matches_slicing():
    """Check that the calculated t0 periods match the t0 periods actually available when slicing"""
    interval_start = minutes(-60)
    interval_end = minutes(180)
    time_resolution = minutes(60)
    max_delay = minutes(90)
    staleness_limit = minutes(180)
    t0_freq = minutes(30)

    # The missing 06:00 initialisation leaves a gap in availability.
    init_times = to_datetime64(["2023-01-02T00:00", "2023-01-02T03:00", "2023-01-02T09:00"])

    steps = np.arange(7) * time_resolution

    da = xr.DataArray(
        np.zeros((len(init_times), len(steps))),
        dims=("init_time_utc", "step"),
        coords={"init_time_utc": init_times, "step": steps},
    )

    # Find the available t0 times for the data
    periods = find_contiguous_t0_periods_nwp(
        init_times=init_times,
        interval_start=interval_start,
        interval_end=interval_end,
        first_forecast_step=steps[0],
        last_forecast_step=steps[-1],
        time_resolution=time_resolution,
        max_delay=max_delay,
        staleness_limit=staleness_limit,
    )
    calculated_t0_times = fill_time_periods(periods, freq=t0_freq)

    # Test slicing out a range of potential t0 times to see if the this agrees with the t0 times
    # calculated above
    potential_t0s = date_range(
        to_datetime64("2023-01-01T22:00"),
        to_datetime64("2023-01-02T12:00"),
        freq=t0_freq,
    )
    sliceable_t0s = []
    for t0 in potential_t0s:
        try:
            sample = select_time_slice_nwp(
                da,
                t0,
                interval_start=interval_start,
                interval_end=interval_end,
                time_resolution=time_resolution,
                delay=max_delay,
            )
        except ValueError:
            continue

        # Slice success alone does not enforce the exclusive staleness limit
        if t0 - sample.init_time_utc.values < staleness_limit:
            sliceable_t0s.append(t0)

    np.testing.assert_array_equal(sliceable_t0s, calculated_t0_times)
