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
    history_duration: pd.Timedelta,
    forecast_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Trim the contiguous time periods to allow for history and forecast durations.

    Args:
        contiguous_time_periods: DataFrame where each row represents a single time period. The 
            DataFrame must have `start_dt` and `end_dt` columns.
        history_duration: Length of the historical slice used for a sample
        forecast_duration: Length of the forecast slice used for a sample


    Returns:
      The contiguous_time_periods DataFrame with the `start_dt` and `end_dt` columns updated.
    """
    contiguous_time_periods = contiguous_time_periods.copy()

    contiguous_time_periods["start_dt"] += history_duration
    contiguous_time_periods["end_dt"] -= forecast_duration

    valid_mask = contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]
    contiguous_time_periods = contiguous_time_periods.loc[valid_mask]

    return contiguous_time_periods



def find_contiguous_t0_periods(
        datetimes: pd.DatetimeIndex,
        history_duration: pd.Timedelta,
        forecast_duration: pd.Timedelta,
        sample_period_duration: pd.Timedelta,
    ) -> pd.DataFrame:
    """Return a pd.DataFrame where each row records the boundary of a contiguous time period.

    Args:
        datetimes: pd.DatetimeIndex. Must be sorted.
        history_duration: Length of the historical slice used for each sample
        forecast_duration: Length of the forecast slice used for each sample
        sample_period_duration: The sample frequency of the timeseries


    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
            has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    total_duration = history_duration + forecast_duration
    
    contiguous_time_periods = find_contiguous_time_periods(
        datetimes=datetimes,
        min_seq_length=int(total_duration / sample_period_duration) + 1,
        max_gap_duration=sample_period_duration,
    )

    contiguous_t0_periods = trim_contiguous_time_periods(
        contiguous_time_periods=contiguous_time_periods,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
    )

    assert len(contiguous_t0_periods) > 0

    return contiguous_t0_periods


def _find_contiguous_t0_periods_nwp(
        ds,
        history_duration: pd.Timedelta,
        forecast_duration: pd.Timedelta,
        max_staleness: pd.Timedelta |  None = None,
        max_dropout: pd.Timedelta = pd.Timedelta(0),
        time_dim: str = "init_time_utc",
        end_buffer: pd.Timedelta = pd.Timedelta(0),
    ):

    assert "step" in ds.coords
    # It is possible to use up to this amount of max staleness for the dataset and slice
    # required
    possible_max_staleness = (
        pd.Timedelta(ds["step"].max().item())
        - forecast_duration
        - end_buffer
    )

    # If max_staleness is set to None we set it based on the max step ahead of the input
    # forecast data
    if max_staleness is None:
        max_staleness = possible_max_staleness
    else:
        # Make sure the max acceptable staleness isn't longer than the max possible
        assert max_staleness <= possible_max_staleness
        max_staleness = max_staleness

    contiguous_time_periods = find_contiguous_t0_periods_nwp(
        datetimes=pd.DatetimeIndex(ds[time_dim]),
        history_duration=history_duration,
        max_staleness=max_staleness,
        max_dropout=max_dropout,
    )
    return contiguous_time_periods



def find_contiguous_t0_periods_nwp(
    datetimes: pd.DatetimeIndex,
    history_duration: pd.Timedelta,
    max_staleness: pd.Timedelta,
    max_dropout: pd.Timedelta = pd.Timedelta(0),
) -> pd.DataFrame:
    """Get all time periods from the NWP init times which are valid as t0 datetimes.

    Args:
        datetimes: Sorted pd.DatetimeIndex
        history_duration: Length of the historical slice used for a sample
        max_staleness: Up to how long after an NWP forecast init_time are we willing to use the
            forecast. Each init time will only be used up to this t0 time regardless of the forecast
            valid time.
        max_dropout: What is the maximum amount of dropout that will be used. This must be <=
            max_staleness.

    Returns:
        pd.DataFrame where each row represents a single time period.  The pd.DataFrame
        has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
    """
    # Sanity checks.
    assert len(datetimes) > 0
    assert datetimes.is_monotonic_increasing
    assert datetimes.is_unique
    assert history_duration >= pd.Timedelta(0)
    assert max_staleness >= pd.Timedelta(0)
    assert max_dropout <= max_staleness

    hist_drop_buffer = max(history_duration, max_dropout)

    # Store contiguous periods
    contiguous_periods = []

    # Start first period allowing for history slice and max dropout
    start_this_period = datetimes[0] + hist_drop_buffer

    # The first forecast is valid up to the max staleness
    end_this_period = datetimes[0] + max_staleness

    for dt_init in datetimes[1:]:
        # If the previous init time becomes stale before the next init becomes valid whilst also
        # considering dropout - then the contiguous period breaks, and new starts with considering
        # dropout and history duration
        if end_this_period < dt_init + max_dropout:
            contiguous_periods.append([start_this_period, end_this_period])

            # And start a new period
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
