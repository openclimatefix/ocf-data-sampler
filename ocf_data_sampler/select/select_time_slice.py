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
    accum_channels: list[str] | None = None,
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
        accum_channels: Channels which are accumulated and need to be differenced
    """
    if accum_channels is None:
        accum_channels = []

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

    # The accumatated and non-accumulated channels
    accum_channels = np.intersect1d(da.channel.values, accum_channels)
    non_accum_channels = np.setdiff1d(da.channel.values, accum_channels)

    start_dt = (t0 + interval_start).ceil(time_resolution)
    end_dt = (t0 + interval_end).ceil(time_resolution)
    target_times = pd.date_range(start_dt, end_dt, freq=time_resolution)

    # Potentially apply NWP dropout
    if consider_dropout and (np.random.uniform() < dropout_frac):
        dt = np.random.choice(dropout_timedeltas)
        t0_available = t0 + dt
    else:
        t0_available = t0

    # Forecasts made up to and including t0
    available_init_times = da.init_time_utc.sel(init_time_utc=slice(None, t0_available))

    # Find the most recent available init times for all target times
    selected_init_times = available_init_times.sel(
        init_time_utc=target_times,
        method="ffill",  # forward fill from init times to target times
    ).values

    # Find the required steps for all target times
    steps = target_times - selected_init_times

    # We want one timestep for each target_time_hourly (obviously!) If we simply do
    # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
    # init_times and steps, which is not what we want! Instead, we use xarray's
    # vectorised-indexing mode via using a DataArray indexer.  See the last example here:
    # https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing

    coords = {"target_time_utc": target_times}
    init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
    step_indexer = xr.DataArray(steps, coords=coords)

    if len(accum_channels) == 0:
        da_sel = da.sel(step=step_indexer, init_time_utc=init_time_indexer)
    else:
        # First minimise the size of the dataset we are diffing
        # - find the init times we are slicing from
        unique_init_times = np.unique(selected_init_times)
        # - find the min and max steps we slice over. Max is extended due to diff
        min_step = min(steps)
        max_step = max(steps) + time_resolution

        da_min = da.sel(init_time_utc=unique_init_times, step=slice(min_step, max_step))

        # Slice out the data which does not need to be diffed
        da_non_accum = da_min.sel(channel=non_accum_channels)
        da_sel_non_accum = da_non_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

        # Slice out the channels which need to be diffed
        da_accum = da_min.sel(channel=accum_channels)

        # Take the diff and slice requested data
        da_accum = da_accum.diff(dim="step", label="lower")
        da_sel_accum = da_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

        # Join diffed and non-diffed variables
        da_sel = xr.concat([da_sel_non_accum, da_sel_accum], dim="channel")

        # Reorder the variable back to the original order
        da_sel = da_sel.sel(channel=da.channel.values)

        # Rename the diffed channels
        da_sel["channel"] = [
            f"diff_{v}" if v in accum_channels else v for v in da_sel.channel.values
        ]

    return da_sel
