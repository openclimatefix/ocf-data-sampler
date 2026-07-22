"""Get contiguous time periods."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.common.time_utils import date_range, datetime_ceil

ZERO_TDELTA = np.timedelta64(0, "ns")


def find_contiguous_time_periods(
    datetimes: NDArray[np.datetime64],
    time_resolution: np.timedelta64,
) -> tuple[NDArray[np.datetime64], NDArray[np.datetime64]]:
    """Return the start and end of all contiguous time periods.

    Args:
        datetimes: Available datetimes - must be sorted.
        time_resolution: The sample frequency of the timeseries.

    Returns:
        A tuple of two NDArray[np.datetime64], where the first array contains the start of each
        contiguous time period and the second array contains the end of each contiguous time period.
    """
    if len(datetimes) == 0:
        raise ValueError("`datetimes` is empty")

    assert_values_unique_increasing(datetimes, "datetimes")

    # Find indices where there are gaps in the datetimes
    gap_mask = np.diff(datetimes) > time_resolution
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into `datetimes` for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into `datetimes`
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the first and last segment of `datetimes`
    segment_boundaries = np.concatenate(([0], segment_boundaries, [len(datetimes)]))

    period_starts = datetimes[segment_boundaries[:-1]]
    period_ends = datetimes[segment_boundaries[1:] - 1]

    return period_starts, period_ends


def find_contiguous_t0_periods(
    datetimes: NDArray[np.datetime64],
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
        datetimes: Available datetimes - must be sorted.
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: The sample frequency of the timeseries

    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
            has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    period_starts, period_ends = find_contiguous_time_periods(
        datetimes=datetimes,
        time_resolution=time_resolution,
    )

    # Keep only periods long enough to contain at least one full sample
    mask = (period_ends - period_starts) >= interval_end - interval_start

    # Shift the boundaries to give the range of valid t0 values in each period
    t0_period_starts = period_starts[mask] - interval_start
    t0_period_ends = period_ends[mask] - interval_end

    if len(t0_period_starts) == 0:
        raise ValueError(
            f"No contiguous time periods found for {datetimes}. "
            f"{interval_start=} {interval_end=} {time_resolution=}",
        )

    return pd.DataFrame({"start_dt": t0_period_starts, "end_dt": t0_period_ends})


def find_contiguous_t0_periods_nwp(
    init_times: NDArray[np.datetime64],
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    first_forecast_step: np.timedelta64,
    last_forecast_step: np.timedelta64,
    max_dropout: np.timedelta64 = ZERO_TDELTA,
    max_staleness: np.timedelta64 | None = None,
) -> pd.DataFrame:
    """Get all time periods from the NWP init-times which are valid as t0 datetimes.

    Args:
        init_times: The initialisation times of the available forecasts.
        interval_start: The start of the time interval with respect to t0.
        interval_end: The end of the time interval with respect to t0.
        first_forecast_step: The timedelta of the first step of the NWP forecast.
        last_forecast_step: The timedelta of the last step of the NWP forecast.
        max_dropout: What is the maximum amount of dropout that will be used.
            This must be <= max_staleness.
        max_staleness: How long after each init-time are we willing to use that init-time. If set to
            None, no additional limit is applied.

    Returns:
        pd.DataFrame where each row represents a single time period. The pd.DataFrame
        has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    assert_values_unique_increasing(init_times, "init_times")

    if len(init_times) == 0:
        raise ValueError("No init-times to use")

    if max_dropout < ZERO_TDELTA:
        raise ValueError("The max dropout must be non-negative (zero or positive)")

    if max_staleness is not None:

        if max_staleness < ZERO_TDELTA:
            raise ValueError("The max staleness must be non-negative (zero or positive)")

        if max_dropout > max_staleness:
            raise ValueError(
                f"max_dropout ({max_dropout}) must be <= max_staleness ({max_staleness})"
            )

        # This is the max staleness we can use considering the max step of the input data
        max_possible_staleness = last_forecast_step - interval_end

        if max_staleness > max_possible_staleness:
            raise ValueError(
                f"max_staleness is too long for the input data, "
                f"{max_staleness=}, {max_possible_staleness=}",
            )

    # We can't use an init-time until this timedelta afterwards to account for dropout
    init_start_timedelta = max(first_forecast_step - interval_start, max_dropout)

    # We can only use an init-time until up to this timedelta afterwards to account for the slice
    # requested and the max_staleness
    if max_staleness is None:
        init_end_timedelta = last_forecast_step - interval_end
    else:
        init_end_timedelta = min(last_forecast_step - interval_end, max_staleness)

    # Store contiguous periods
    contiguous_periods: list[list[np.datetime64]] = []

    # This is the range of t0 times available whilst using the first init-time
    start_this_period = init_times[0] + init_start_timedelta
    end_this_period = init_times[0] + init_end_timedelta

    for init_time in init_times[1:]:
        # If the previous init-time doesn't cover t0 times up to when this init-time covers them
        # from, then we break the contiguous period
        if end_this_period < init_time + init_start_timedelta:
            contiguous_periods.append([start_this_period, end_this_period])
            # The new period begins with the same conditions as the first period
            start_this_period = init_time + init_start_timedelta
        end_this_period = init_time + init_end_timedelta

    contiguous_periods.append([start_this_period, end_this_period])

    return pd.DataFrame(contiguous_periods, columns=["start_dt", "end_dt"])


def intersect_time_periods(time_periods: list[pd.DataFrame]) -> pd.DataFrame:
    """Find the intersection of list of time periods.

    Consecutively updates intersection of time periods.
    See the docstring of _intersect_2_time_periods() for further details.
    """
    if len(time_periods) == 0:
        raise ValueError("No time periods to intersect")

    for i, periods in enumerate(time_periods):
        if periods.empty:
            raise ValueError(f"Time period frame {i} contains no periods")

    intersection = time_periods[0]
    for periods in time_periods[1:]:
        intersection = _intersect_2_time_periods(intersection, periods)
        if intersection.empty:
            return intersection
    return intersection


def _intersect_2_time_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of two pd.DataFrames of time periods.

    Each row of each pd.DataFrame represents a single time period.  Each pd.DataFrame has
    two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

    Graphical representation of two pd.DataFrames of time periods and their intersection,
    as follows:

                  ---------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a: pd.DataFrame where each row represents a time period. The pd.DataFrame has
        two columns: start_dt and end_dt.
        b: pd.DataFrame where each row represents a time period. The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        The intersecting time periods, sorted by start time, as a pd.DataFrame with two
        columns: start_dt and end_dt. Empty if no periods overlap.
    """
    if a.empty:
        raise ValueError("Input `a` contains no periods")
    if b.empty:
        raise ValueError("Input `b` contains no periods")

    # Maybe switch these for efficiency in the next section. We will do the native python loop over
    # the shorter dataframe
    if len(a) > len(b):
        a, b = b, a

    a_starts = a["start_dt"].values
    a_ends = a["end_dt"].values

    b_starts = b["start_dt"].values
    b_ends = b["end_dt"].values

    all_starts: list[NDArray[np.datetime64]] = []
    all_ends: list[NDArray[np.datetime64]] = []

    for i in range(len(a)):

        # The overlapping periods can't start before either period starts, and can't end after
        # either period ends. So we take the max of the start times and the min of the end times.
        starts = np.maximum(a_starts[i], b_starts)
        ends = np.minimum(a_ends[i], b_ends)

        # Any periods that don't overlap will have a start time that is after the end time.
        # We filter those out.
        overlap_mask = starts <= ends
        starts = starts[overlap_mask]
        ends = ends[overlap_mask]

        all_starts.append(starts)
        all_ends.append(ends)

    intersecting_periods = pd.DataFrame({
        "start_dt": np.concatenate(all_starts),
        "end_dt": np.concatenate(all_ends),
    })
    return intersecting_periods.sort_values(by="start_dt").reset_index(drop=True)


def fill_time_periods(time_periods: pd.DataFrame, freq: np.timedelta64) -> NDArray[np.datetime64]:
    """Create range of timestamps between given start and end times.

    Each of the continuous periods (i.e. each row of the input DataFrame) is filled with the
    specified frequency.

    Args:
        time_periods: DataFrame with columns 'start_dt' and 'end_dt'
        freq: Frequency to fill time periods with
    """
    start_dts = datetime_ceil(time_periods["start_dt"].values, freq)
    end_dts = time_periods["end_dt"].values
    date_ranges = [
        date_range(start_dt, end_dt, freq=freq)
        for start_dt, end_dt in zip(start_dts, end_dts, strict=True)
    ]
    return np.unique(np.concatenate(date_ranges))
