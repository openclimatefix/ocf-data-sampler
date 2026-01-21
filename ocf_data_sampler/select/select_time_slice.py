"""Select a time slice from a Dataset or DataArray."""

import numpy as np
import pandas as pd
import xarray as xr


def select_time_slice(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    time_resolution: pd.Timedelta,
) -> xr.DataArray:
    """Select a time slice from a DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
    """
    start_dt = t0 + interval_start
    end_dt = t0 + interval_end

    start_dt = start_dt.ceil(time_resolution)
    end_dt = end_dt.ceil(time_resolution)

    return da.sel(time_utc=slice(start_dt, end_dt))


def select_time_slice_nwp(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    time_resolution: pd.Timedelta,
    dropout_timedeltas: list[pd.Timedelta] | None = None,
    dropout_frac: float | None = 0,
) -> xr.DataArray:
    """Select a time slice from an NWP DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
        dropout_timedeltas: List of possible timedeltas before t0 where data availability may start
        dropout_frac: Probability to apply dropout
    """
    # Input checking
    if dropout_timedeltas is None:
        dropout_timedeltas = []

    if len(dropout_timedeltas) > 0:
        if not all(t < pd.Timedelta(0) for t in dropout_timedeltas):
            raise ValueError("dropout timedeltas must be negative")
        if len(dropout_timedeltas) < 1:
            raise ValueError("dropout timedeltas must have at least one element")

    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1")

    consider_dropout = len(dropout_timedeltas) > 0 and dropout_frac > 0

    start_dt = (t0 + interval_start).ceil(time_resolution)
    end_dt = (t0 + interval_end).ceil(time_resolution)
    target_times = pd.date_range(start_dt, end_dt, freq=time_resolution)

    # Potentially apply NWP dropout
    if consider_dropout and (np.random.uniform() < dropout_frac):
        t0_available = t0 + np.random.choice(dropout_timedeltas)
    else:
        t0_available = t0

    # Find the window of all possible `init_time`s whose forecast horizons could cover the
    # start of the target period. This correctly handles the case where the requested time range
    # does not contain an `init_time` itself, but is covered by a forecast from a
    # previous `init_time`.
    #
    # For example, if the last NWP init_time was 12:00 with a 36-hour forecast, and we
    # request data for 14:00-18:00, this logic will correctly identify the 12:00 init_time
    # as a valid source.
    t_min = target_times[0] - da.step.values[-1]
    init_times = da.init_time_utc.values
    available_init_times = init_times[(t_min <= init_times) & (init_times <= t0_available)]

    # Check if there are any available init times
    if len(available_init_times) == 0:
        max_step = da.step.values[-1]
        raise ValueError(
            f"Cannot get NWP data for target time {target_times[0]}. "
            f"The latest available init_time is {init_times[-1]}, but an init_time of at least "
            f"{t_min} is required to cover this target time (given a "
            f"maximum forecast horizon of {max_step}).",
        )

    # Use numpy.searchsorted to find the index of the most recent available init-time for each
    # target-time. `side="right"` ensures that if a target_time is identical to an init_time,
    # we get the index of the init_time itself. Subtracting 1 then gives us the index of the
    # latest init_time that is less than or equal to the target_time.
    indices = np.searchsorted(available_init_times, target_times.values, side="right")
    selected_indices = indices - 1

    # Check for indices less than 0, which indicate a target_time was before the first available
    if np.any(selected_indices < 0):
        first_bad_idx = np.where(selected_indices < 0)[0][0]
        first_bad_target = target_times[first_bad_idx]
        raise ValueError(
            f"Target time {first_bad_target} is before the first available init time"
            f" {available_init_times[0]}.",
        )

    selected_init_times = available_init_times[selected_indices]

    # Find the required steps for all target-times
    steps = target_times - selected_init_times

    # If we are only selecting from one init-time we can construct the slice so its faster
    if len(np.unique(selected_init_times)) == 1:
        da_sel = da.sel(init_time_utc=selected_init_times[0], step=slice(steps[0], steps[-1]))

    # If we are selecting from multiple init times this more complex and slower
    else:
        # We want one timestep for each target_time. If we simply do
        # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
        # init_times and steps, which is not what we want! Instead, we use xarray's
        # vectorised-indexing mode via using a DataArray indexer. See the last example here:
        # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
        coords = {"time_utc": target_times}
        init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
        step_indexer = xr.DataArray(steps, coords=coords)
        da_sel = da.sel(init_time_utc=init_time_indexer, step=step_indexer)

    return da_sel
