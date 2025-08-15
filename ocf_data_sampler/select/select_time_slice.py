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

    if len(dropout_timedeltas)>0:
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

    # Get the available and relevant init-times
    t_min = target_times[0] - da.step.values[-1]
    init_times = da.init_time_utc.values
    available_init_times = init_times[(t_min<=init_times) & (init_times<=t0_available)]

    # Find the most recent available init-times for all target-times
    selected_init_times = np.array(
        [available_init_times[available_init_times<=t][-1] for t in target_times],
    )

    # Find the required steps for all target-times
    steps = target_times - selected_init_times

    # If we are only selecting from one init-time we can construct the slice so its faster
    if len(np.unique(selected_init_times))==1:
        da_sel = da.sel(init_time_utc=selected_init_times[0], step=slice(steps[0], steps[-1]))

    # If we are selecting from multiple init times this more complex and slower
    else:
        # We want one timestep for each target_time_hourly (obviously!) If we simply do
        # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
        # init_times and steps, which is not what we want! Instead, we use xarray's
        # vectorised-indexing mode via using a DataArray indexer.  See the last example here:
        # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing
        coords = {"step": steps}
        init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
        step_indexer = xr.DataArray(steps, coords=coords)
        da_sel = da.sel(init_time_utc=init_time_indexer, step=step_indexer)

    return da_sel
