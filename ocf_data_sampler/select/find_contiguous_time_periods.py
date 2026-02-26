"""Get contiguous time periods."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ocf_data_sampler.load.utils import assert_values_unique_increasing

ZERO_TDELTA = np.timedelta64(0, "ns")


def find_contiguous_time_periods(
    datetimes: NDArray[np.datetime64],
    min_seq_length: int,
    max_gap_duration: np.timedelta64,
) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
      datetimes: Available datetimes - must be sorted.
      min_seq_length: Sequences of min_seq_length or shorter will be discarded.
      max_gap_duration: If any pair of consecutive `datetimes` is more than `max_gap_duration`
        apart, then this pair of `datetimes` will be considered a "gap" between two contiguous
        sequences.

    Returns:
      pd.DataFrame where each row represents a single time period. The pd.DataFrame
      has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    if len(datetimes) == 0:
        raise ValueError("No datetimes to use")
    if min_seq_length <= 1:
        raise ValueError(f"{min_seq_length=} must be greater than 1")

    assert_values_unique_increasing(datetimes, "datetimes")

    # Find indices of gaps larger than max_gap:
    gap_mask = np.diff(datetimes) > max_gap_duration
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    # So we add 1 to gap_indices to get segment_boundaries, an index into dt_index
    # which identifies the _start_ of each segment.
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(datetimes)]))

    periods: list[list[np.datetime64]] = []
    start_i = 0
    for next_start_i in segment_boundaries:
        n_timesteps = next_start_i - start_i
        if n_timesteps > min_seq_length:
            end_i = next_start_i - 1
            periods.append([datetimes[start_i], datetimes[end_i]])
        start_i = next_start_i

    if len(periods) == 0:
        raise ValueError(
            f"Did not find any periods from {datetimes}. {min_seq_length=} {max_gap_duration=}",
        )

    return pd.DataFrame(periods, columns=["start_dt", "end_dt"])


def trim_contiguous_time_periods(
    contiguous_time_periods: pd.DataFrame,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
) -> pd.DataFrame:
    """Trims contiguous time periods to account for history requirements and forecast horizons.

    Args:
        contiguous_time_periods: pd.DataFrame where each row represents a single time period.
            The pd.DataFrame must have `start_dt` and `end_dt` columns.
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0

    Returns:
      The contiguous_time_periods pd.DataFrame with the `start_dt` and `end_dt` columns updated.
    """
    # Make a copy so the data is not edited in place.
    trimmed_time_periods = contiguous_time_periods.copy()
    trimmed_time_periods["start_dt"] -= interval_start
    trimmed_time_periods["end_dt"] -= interval_end

    valid_mask = trimmed_time_periods["start_dt"] <= trimmed_time_periods["end_dt"]

    return trimmed_time_periods.loc[valid_mask]


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
    assert_values_unique_increasing(datetimes, "datetimes")

    total_duration = interval_end - interval_start

    contiguous_time_periods = find_contiguous_time_periods(
        datetimes=datetimes,
        min_seq_length=int(total_duration / time_resolution) + 1,
        max_gap_duration=time_resolution,
    )

    contiguous_t0_periods = trim_contiguous_time_periods(
        contiguous_time_periods=contiguous_time_periods,
        interval_start=interval_start,
        interval_end=interval_end,
    )

    if len(contiguous_t0_periods) == 0:
        raise ValueError(
            f"No contiguous time periods found for {datetimes}. "
            f"{interval_start=} {interval_end=} {time_resolution=}",
        )

    return contiguous_t0_periods


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
        raise ValueError("The max dropout must be positive")

    if max_staleness is not None:

        if max_staleness < ZERO_TDELTA:
            raise ValueError("The max staleness must be positive")

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


def intersection_of_multiple_dataframes_of_periods(
    time_periods: list[pd.DataFrame],
) -> pd.DataFrame:
    """Find the intersection of list of time periods.

    Consecutively updates intersection of time periods.
    See the docstring of intersection_of_2_dataframes_of_periods() for further details.
    """
    if len(time_periods) == 0:
        raise ValueError("No time periods to intersect")
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

    Graphical representation of two pd.DataFrames of time periods and their intersection,
    as follows:

                 ----------------------> TIME ->---------------------
               a: |-----|   |----|     |----------|     |-----------|
               b:    |--------|                       |----|    |---|
    intersection:    |--|   |-|                         |--|    |---|

    Args:
        a: pd.DataFrame where each row represents a time period. The pd.DataFrame has
        two columns: start_dt and end_dt.
        b: pd.DataFrame where each row represents a time period. The pd.DataFrame has
        two columns: start_dt and end_dt.

    Returns:
        Sorted list of intersecting time periods represented as a pd.DataFrame with two columns:
        start_dt and end_dt.
    """
    if a.empty or b.empty:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    # Maybe switch these for efficiency in the next section. We will do the native python loop over
    # the shorter dataframe
    if len(a) > len(b):
        a, b = b, a

    all_intersecting_periods = []
    for a_period in a.itertuples():
        # Five ways in which two periods may overlap:
        # a: |----| or |---|   or  |---| or   |--|   or |-|
        # b:  |--|       |---|   |---|      |------|    |-|
        # In all five, `a` must always start before (or equal to) where `b` ends,
        # and `a` must always end after (or equal to) where `b` starts.

        # There are two ways in which two periods may *not* overlap:
        # a: |---|        or        |---|
        # b:       |---|      |---|
        # `overlapping_periods` will not include periods which do *not* overlap.

        overlapping_periods = b[(a_period.start_dt <= b.end_dt) & (a_period.end_dt >= b.start_dt)]

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
