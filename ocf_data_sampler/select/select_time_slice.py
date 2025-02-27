"""Select a time slice from a Dataset or DataArray."""

import numpy as np
import pandas as pd
import xarray as xr


def select_time_slice(
    ds: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    sample_period_duration: pd.Timedelta,
) -> xr.DataArray:
    """Select a time slice from a Dataset or DataArray."""
    t0_datetime_utc = pd.Timestamp(t0)
    start_dt = t0_datetime_utc + interval_start
    end_dt = t0_datetime_utc + interval_end

    start_dt = start_dt.ceil(sample_period_duration)
    end_dt = end_dt.ceil(sample_period_duration)

    return ds.sel(time_utc=slice(start_dt, end_dt))


def select_time_slice_nwp(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    sample_period_duration: pd.Timedelta,
    dropout_timedeltas: list[pd.Timedelta] | None = None,
    dropout_frac: float | None = 0,
    accum_channels: list[str] | None = None,
    channel_dim_name: str = "channel",
) -> xr.DataArray:
    """Select a time slice from a NWP DataArray."""
    if accum_channels is None:
        accum_channels = []
    if dropout_timedeltas is not None:
        if not all(t < pd.Timedelta(0) for t in dropout_timedeltas):
            raise ValueError("dropout timedeltas must be negative")
        if len(dropout_timedeltas) < 1:
            raise ValueError("dropout timedeltas must have at least one element")
    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1")
    consider_dropout = (dropout_timedeltas is not None) and dropout_frac > 0

    # The accumatation and non-accumulation channels
    accum_channels = np.intersect1d(
        da[channel_dim_name].values,
        accum_channels,
    )
    non_accum_channels = np.setdiff1d(
        da[channel_dim_name].values,
        accum_channels,
    )

    start_dt = (t0 + interval_start).ceil(sample_period_duration)
    end_dt = (t0 + interval_end).ceil(sample_period_duration)

    target_times = pd.date_range(start_dt, end_dt, freq=sample_period_duration)

    # Maybe apply NWP dropout
    if consider_dropout and (np.random.uniform() < dropout_frac):
        dt = np.random.choice(dropout_timedeltas)
        t0_available = t0 + dt
    else:
        t0_available = t0

    # Forecasts made up to and including t0
    available_init_times = da.init_time_utc.sel(
        init_time_utc=slice(None, t0_available),
    )

    # Find the most recent available init times for all target times
    selected_init_times = available_init_times.sel(
        init_time_utc=target_times,
        method="ffill",  # forward fill from init times to target times
    ).values

    # Find the required steps for all target times
    steps = target_times - selected_init_times

    # We want one timestep for each target_time_hourly (obviously!) If we simply do
    # nwp.sel(init_time=init_times, step=steps) then we'll get the *product* of
    # init_times and steps, which is not what # we want! Instead, we use xarray's
    # vectorized-indexing mode by using a DataArray indexer.  See the last example here:
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
        max_step = max(steps) + sample_period_duration

        da_min = da.sel(
            {
                "init_time_utc": unique_init_times,
                "step": slice(min_step, max_step),
            },
        )

        # Slice out the data which does not need to be diffed
        da_non_accum = da_min.sel({channel_dim_name: non_accum_channels})
        da_sel_non_accum = da_non_accum.sel(
            step=step_indexer,
            init_time_utc=init_time_indexer,
        )

        # Slice out the channels which need to be diffed
        da_accum = da_min.sel({channel_dim_name: accum_channels})

        # Take the diff and slice requested data
        da_accum = da_accum.diff(dim="step", label="lower")
        da_sel_accum = da_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

        # Join diffed and non-diffed variables
        da_sel = xr.concat([da_sel_non_accum, da_sel_accum], dim=channel_dim_name)

        # Reorder the variable back to the original order
        da_sel = da_sel.sel({channel_dim_name: da[channel_dim_name].values})

        # Rename the diffed channels
        da_sel[channel_dim_name] = [
            f"diff_{v}" if v in accum_channels else v for v in da_sel[channel_dim_name].values
        ]

    return da_sel
