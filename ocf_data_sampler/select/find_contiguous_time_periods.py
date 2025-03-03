"""Get contiguous time periods for training."""

import numpy as np
import pandas as pd

from ocf_data_sampler.load.utils import check_time_unique_increasing

ZERO_TDELTA = pd.Timedelta(0)


def find_contiguous_time_periods(
    datetimes: pd.DatetimeIndex,
    min_seq_length: int,
    max_gap_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
      datetimes: pd.DatetimeIndex. Must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.  Typically, this
        would be set to the `total_seq_length` of each machine learning example.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences. Typically, `max_gap_duration` would be set to the sample period of
        the timeseries.

    Returns:
      pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    if len(datetimes) == 0:
        raise ValueError("No datetimes to use")
    if min_seq_length < 1:
        raise ValueError(f"{min_seq_length=} must be >= 1")
    check_time_unique_increasing(datetimes)

    # Handle single timestamp case
    if len(datetimes) == 1:
        if min_seq_length == 1:
            return pd.DataFrame([{"start_dt": datetimes[0], "end_dt": datetimes[0]}])
        else:
            raise ValueError(
                "Only one timestamp found, but min_seq_length > 1. No valid periods.",
            )

    # Find indices of gaps larger than max_gap:
    gap_mask = pd.TimedeltaIndex(np.diff(datetimes)) > max_gap_duration
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into dt_index
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    periods: list[dict[str, pd.Timestamp]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps >= min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    if len(periods) == 0:
        raise ValueError(
            f"Did not find any periods from {datetimes}. {min_seq_length=} {max_gap_duration=}",
        )

    return pd.DataFrame(periods)


def trim_contiguous_time_periods(
    contiguous_time_periods: pd.DataFrame,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
) -> pd.DataFrame:
    """Trim the contiguous time periods to allow for history and forecast durations.

    Args:
        contiguous_time_periods: DataFrame where each row represents a single time period.
            The DataFrame must have `start_dt` and `end_dt` columns.
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0


    Returns:
      The contiguous_time_periods DataFrame with the `start_dt` and `end_dt` columns updated.
    """
    contiguous_time_periods = contiguous_time_periods.copy()

    contiguous_time_periods["start_dt"] -= interval_start
    contiguous_time_periods["end_dt"] -= interval_end

    valid_mask = contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]
    contiguous_time_periods = contiguous_time_periods.loc[valid_mask]

    return contiguous_time_periods


def find_contiguous_t0_periods(
    datetimes: pd.DatetimeIndex,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    sample_period_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
        datetimes: pd.DatetimeIndex. Must be sorted.
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        sample_period_duration: The sample frequency of the timeseries


    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
            has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    total_duration = interval_end - interval_start

    contiguous_time_periods = find_contiguous_time_periods(
        datetimes=datetimes,
        min_seq_length=int(total_duration / sample_period_duration) + 1,
        max_gap_duration=sample_period_duration,
    )

    contiguous_t0_periods = trim_contiguous_time_periods(
        contiguous_time_periods=contiguous_time_periods,
        interval_start=interval_start,
        interval_end=interval_end,
    )

    if len(contiguous_t0_periods) == 0:
        raise ValueError(
            f"No contiguous time periods found for {datetimes}. "
            f"{interval_start=} {interval_end=} {sample_period_duration=}",
        )

    return contiguous_t0_periods


def find_contiguous_t0_periods_nwp(
    init_times: pd.DatetimeIndex,
    interval_start: pd.Timedelta,
    max_staleness: pd.Timedelta,
    max_dropout: pd.Timedelta = ZERO_TDELTA,
    first_forecast_step: pd.Timedelta = ZERO_TDELTA,
) -> pd.DataFrame:
    """Get all time periods from the NWP init times which are valid as t0 datetimes.

    Args:
        init_times: The initialisation times of the available forecasts
        interval_start: The start of the desired data interval with respect to t0
        max_staleness: Up to how long after an init time are we willing to use the forecast.
            Each init time will only be used up to this t0 time
            regardless of the forecast valid time.
        max_dropout: What is the maximum amount of dropout that will be used.
            This must be <= max_staleness.
        first_forecast_step: The timedelta of the first step of the forecast.
            By default we assume the first valid time of the forecast
            is the same as its init time.

    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
        has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    if len(init_times) == 0:
        raise ValueError("No init times to use")
    if not init_times.is_monotonic_increasing:
        raise ValueError("Init times must be sorted and monotinically increasing")
    if not init_times.is_unique:
        raise ValueError("Init times must be unique")
    if max_staleness < pd.Timedelta(0):
        raise ValueError("The max staleness must be positive")
    if not (pd.Timedelta(0) <= max_dropout <= max_staleness):
        raise ValueError("The max dropout must be between 0 and the max staleness")

    hist_drop_buffer = max(first_forecast_step - interval_start, max_dropout)

    # Store contiguous periods
    contiguous_periods = []

    # Begin the first period allowing for the time to the first_forecast_step, the length of the
    # interval sampled from before t0, and the dropout
    start_this_period = init_times[0] + hist_drop_buffer

    # The first forecast is valid up to the max staleness
    end_this_period = init_times[0] + max_staleness

    for dt_init in init_times[1:]:
        # If the previous init time becomes stale before the next init becomes valid (whilst also
        # considering dropout) then the contiguous period breaks
        # Else if the previous init time becomes stale before the fist step of the next forecast
        # then this also causes a break in the contiguous period
        if end_this_period < dt_init + max(max_dropout, first_forecast_step):
            contiguous_periods.append([start_this_period, end_this_period])
            # The new period begins with the same conditions as the first period
            start_this_period = dt_init + hist_drop_buffer
        end_this_period = dt_init + max_staleness

    contiguous_periods.append([start_this_period, end_this_period])

    return pd.DataFrame(contiguous_periods, columns=["start_dt", "end_dt"])


def intersection_of_multiple_dataframes_of_periods(
    time_periods: list[pd.DataFrame],
) -> pd.DataFrame:
    """Find the intersection of a list of time periods.

    See the docstring of intersection_of_2_dataframes_of_periods() for more details.
    """
    if len(time_periods) == 0:
        raise ValueError("No time periods to intersect")
    intersection = time_periods[0]
    for time_period in time_periods[1:]:
        intersection = intersection_of_2_dataframes_of_periods(intersection, time_period)
    return intersection


def intersection_of_2_dataframes_of_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of two DataFrames of time periods, allowing an overlap of length 1.

    Intervals are treated as inclusive of both endpoints.
    """
    # Intersection of two dataframes can be as follows:

    # 🟩 Condition 1: a fully contains b
    # Diagram:
    #   a: ┌───────┐
    #   b:     └───┘
    # Explanation: Interval 'a' fully contains interval 'b'. The intersection is the entirety of 'b'

    # 🟩 Condition 2: b fully contains a
    # Diagram:
    #   a:    ┌───┐
    #   b: ┌─────────┐
    # Explanation: Interval 'b' fully contains interval 'a'. The intersection is the entirety of 'a'

    # 🟩 Condition 3: Overlap at the start of a
    # Diagram:
    #   a:    ┌──────┐
    #   b: ┌────┘
    # Explanation: Interval 'b' starts inside interval 'a'. The intersection starts from 'b's start
    # and ends at 'a's end.

    # 🟩 Condition 4: Overlap at the start of b
    # Diagram:
    #   a: ┌──────┐
    #   b:    └──────┐
    # Explanation: Interval 'a' starts before 'b' but they overlap. The intersection starts from
    # 'b's start and ends at 'a's end.

    # 🟩 Condition 5: Overlap at the end of a
    # Diagram:
    #   a: ┌────┐
    #    b:     └────┐
    # Explanation: Interval 'b' ends inside interval 'a'. The intersection starts at 'b's start
    # and ends at 'a's end.

    # 🟩 Condition 6: Exact match
    # Diagram:
    #   a: ┌──────┐
    #   b: ┌──────┐
    # Explanation: Intervals 'a' and 'b' are exactly the same. The intersection is the
    # entirety of both intervals.

    # 🟩 Condition 7: Single point overlap
    # Diagram:
    #   a: ┌──────┐
    #   b:   └┐
    # OR
    #   b:   ┌┘
    #   a: └──────┘
    # Explanation: There is only a single point of overlap where the end of one interval
    # touches the start of the other.

    # 🟩 Condition 8: No overlap (Single point gap)
    # Diagram:
    #   a: ┌──────┐
    #       (gap)
    #   b:      ┌──────┐
    # Explanation: There is no overlap, as the intervals are completely separated by a gap.
    # Expected intersection: Empty DataFrame (No valid overlap)

    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    # Create a Cartesian product (cross join) of all rows in a and b.
    merged = pd.merge(a, b, how="cross")  # Columns: start_dt_x, end_dt_x, start_dt_y, end_dt_y

    # Use inclusive conditions for overlapping:
    # Two intervals overlap if a.start <= b.end AND b.start <= a.end.
    merged = merged[
        (merged["start_dt_x"] <= merged["end_dt_y"]) & (merged["end_dt_x"] >= merged["start_dt_y"])
    ]

    # Calculate the intersection:
    # The intersection starts at the later start date and ends at the earlier end date.
    merged["start_dt"] = merged[["start_dt_x", "start_dt_y"]].max(axis=1)
    merged["end_dt"] = merged[["end_dt_x", "end_dt_y"]].min(axis=1)

    # Ensure the computed intersection is valid (start_dt should not be after end_dt).
    # (If they are equal, it's a single-point overlap which is allowed.)
    valid = merged["start_dt"] <= merged["end_dt"]
    merged = merged[valid]

    # Select only the relevant columns, remove any duplicate intersections,
    # sort by start_dt, and reset the index.
    result = (
        merged[["start_dt", "end_dt"]]
        .drop_duplicates()
        .sort_values(by="start_dt")
        .reset_index(drop=True)
    )
    return result
