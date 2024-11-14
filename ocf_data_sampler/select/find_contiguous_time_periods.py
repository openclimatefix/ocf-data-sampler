"""Get contiguous time periods for training"""

import numpy as np
import pandas as pd



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
    assert len(datetimes) > 0
    assert min_seq_length > 1
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique

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
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            period = {"start_dt": datetimes[start_i], "end_dt": datetimes[end_i]}
            periods.append(period)
        start_i = next_start_i

    assert len(periods) > 0, (
        f"Did not find an periods from {datetimes}. " f"{min_seq_length=} {max_gap_duration=}"
    )

    return pd.DataFrame(periods)


def trim_contiguous_time_periods(
    contiguous_time_periods: pd.DataFrame, 
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
) -> pd.DataFrame:
    """Trim the contiguous time periods to allow for history and forecast durations.

    Args:
        contiguous_time_periods: DataFrame where each row represents a single time period. The 
            DataFrame must have `start_dt` and `end_dt` columns.
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

    assert len(contiguous_t0_periods) > 0

    return contiguous_t0_periods


def find_contiguous_t0_periods_nwp(
    init_times: pd.DatetimeIndex,
    interval_start: pd.Timedelta,
    max_staleness: pd.Timedelta,
    max_dropout: pd.Timedelta = pd.Timedelta(0),
    first_forecast_step: pd.Timedelta = pd.Timedelta(0),
    
) -> pd.DataFrame:
    """Get all time periods from the NWP init times which are valid as t0 datetimes.

    Args:
        init_times: The initialisation times of the available forecasts
        interval_start: The start of the desired data interval with respect to t0
        max_staleness: Up to how long after an init time are we willing to use the forecast. Each 
            init time will only be used up to this t0 time regardless of the forecast valid time.
        max_dropout: What is the maximum amount of dropout that will be used. This must be <=
            max_staleness.
        first_forecast_step: The timedelta of the first step of the forecast. By default we assume
            the first valid time of the forecast is the same as its init time.

    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
        has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(init_times) > 0
    assert init_times.is_monotonic_increasing
    assert init_times.is_unique
    assert max_staleness >= pd.Timedelta(0)
    assert pd.Timedelta(0) <= max_dropout <= max_staleness

    hist_drop_buffer = max(first_forecast_step-interval_start, max_dropout)

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
        if (end_this_period < dt_init + max(max_dropout, first_forecast_step)):
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
    assert len(time_periods) > 0
    intersection = time_periods[0]
    for time_period in time_periods[1:]:
        intersection = intersection_of_2_dataframes_of_periods(intersection, time_period)
    return intersection


def intersection_of_2_dataframes_of_periods(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Find the intersection of two pd.DataFrames of time periods.

    Each row of each pd.DataFrame represents a single time period.  Each pd.DataFrame has
    two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

    A typical use-case is that each pd.DataFrame represents all the time periods where
    a `DataSource` has contiguous, valid data.

    Here's a graphical example of two pd.DataFrames of time periods and their intersection:

                 ----------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.
        b: pd.DataFrame where each row represents a time period.  The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        Sorted list of intersecting time periods represented as a pd.DataFrame with two columns:
        start_dt and end_dt.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    all_intersecting_periods = []
    for a_period in a.itertuples():
        # Five ways in which two periods may overlap:
        # a: |----| or |---|   or  |---| or   |--|   or |-|
        # b:  |--|       |---|   |---|      |------|    |-|
        # In all five, `a` must always start before `b` ends,
        # and `a` must always end after `b` starts:

        # TODO: <= and >= because we should allow overlap time periods of length 1. e.g.
        # a: |----|      or   |---|   
        # b:      |--|            |---|
        # These aren't allowed if we use < and >.

        overlapping_periods = b[(a_period.start_dt < b.end_dt) & (a_period.end_dt > b.start_dt)]

        # There are two ways in which two periods may *not* overlap:
        # a: |---|        or        |---|
        # b:       |---|      |---|
        # `overlapping` will not include periods which do *not* overlap.

        # Now find the intersection of each period in `overlapping_periods` with
        # the period from `a` that starts at `a_start_dt` and ends at `a_end_dt`.
        # We do this by clipping each row of `overlapping_periods`
        # to start no earlier than `a_start_dt`, and end no later than `a_end_dt`.

        # First, make a copy, so we don't clip the underlying data in `b`.
        intersection = overlapping_periods.copy()
        intersection["start_dt"] = intersection.start_dt.clip(lower=a_period.start_dt)
        intersection["end_dt"] = intersection.end_dt.clip(upper=a_period.end_dt)

        all_intersecting_periods.append(intersection)

    all_intersecting_periods = pd.concat(all_intersecting_periods)
    return all_intersecting_periods.sort_values(by="start_dt").reset_index(drop=True)
