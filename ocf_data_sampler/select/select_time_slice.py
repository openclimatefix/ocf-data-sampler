import xarray as xr
import pandas as pd
import numpy as np



def _sel_fillnan(
        da: xr.DataArray, 
        start_dt: pd.Timestamp, 
        end_dt: pd.Timestamp, 
        sample_period_duration: pd.Timedelta,
    ) -> xr.DataArray:
    """Select a time slice from a DataArray, filling missing times with NaNs."""
    requested_times = pd.date_range(start_dt, end_dt, freq=sample_period_duration)
    return da.reindex(time_utc=requested_times)


def _sel_default(
        da: xr.DataArray, 
        start_dt: pd.Timestamp, 
        end_dt: pd.Timestamp, 
        sample_period_duration: pd.Timedelta,
    ) -> xr.DataArray:
    """Select a time slice from a DataArray, without filling missing times."""
    return da.sel(time_utc=slice(start_dt, end_dt))


# TODO either implement this or remove it, which would tidy up the code
def _sel_fillinterp(
        da: xr.DataArray, 
        start_dt: pd.Timestamp, 
        end_dt: pd.Timestamp, 
        sample_period_duration: pd.Timedelta,
    ) -> xr.DataArray:
    """Select a time slice from a DataArray, filling missing times with linear interpolation."""
    return NotImplemented


def select_time_slice(
    ds: xr.DataArray,
    t0: pd.Timestamp,
    sample_period_duration: pd.Timedelta,
    history_duration: pd.Timedelta | None = None,
    forecast_duration: pd.Timedelta | None = None,
    interval_start: pd.Timedelta | None = None,
    interval_end: pd.Timedelta | None = None,
    fill_selection: bool = False,
    max_steps_gap: int = 0,
):
    """Select a time slice from a Dataset or DataArray."""
    used_duration = history_duration is not None and forecast_duration is not None
    used_intervals = interval_start is not None and interval_end is not None
    assert used_duration ^ used_intervals, "Either durations, or intervals must be supplied"
    assert max_steps_gap >= 0, "max_steps_gap must be >= 0 "

    if used_duration:
        interval_start = - history_duration
        interval_end = forecast_duration
    
    if fill_selection and max_steps_gap == 0:
        _sel = _sel_fillnan
    elif fill_selection and max_steps_gap > 0:
        _sel = _sel_fillinterp
    else:
        _sel = _sel_default

    t0_datetime_utc = pd.Timestamp(t0)
    start_dt = t0_datetime_utc + interval_start
    end_dt = t0_datetime_utc + interval_end

    start_dt = start_dt.ceil(sample_period_duration)
    end_dt = end_dt.ceil(sample_period_duration)

    return _sel(ds, start_dt, end_dt, sample_period_duration)


def select_time_slice_nwp(
    ds: xr.DataArray,
    t0: pd.Timestamp,
    sample_period_duration: pd.Timedelta,
    history_duration: pd.Timedelta,
    forecast_duration: pd.Timedelta,
    dropout_timedeltas: list[pd.Timedelta] | None = None,
    dropout_frac: float | None = 0,
    accum_channels: list[str] = [],
    channel_dim_name: str = "channel",
):

    if dropout_timedeltas is not None:
        assert all(
            [t < pd.Timedelta(0) for t in dropout_timedeltas]
        ), "dropout timedeltas must be negative"
        assert len(dropout_timedeltas) >= 1
    assert 0 <= dropout_frac <= 1
    _consider_dropout = (dropout_timedeltas is not None) and dropout_frac > 0


    # The accumatation and non-accumulation channels
    accum_channels = np.intersect1d(
        ds[channel_dim_name].values, accum_channels
    )
    non_accum_channels = np.setdiff1d(
        ds[channel_dim_name].values, accum_channels
    )

    start_dt = (t0 - history_duration).ceil(sample_period_duration)
    end_dt = (t0 + forecast_duration).ceil(sample_period_duration)

    target_times = pd.date_range(start_dt, end_dt, freq=sample_period_duration)

    # Maybe apply NWP dropout
    if _consider_dropout and (np.random.uniform() < dropout_frac):
        dt = np.random.choice(dropout_timedeltas)
        t0_available = t0 + dt
    else:
        t0_available = t0

    # Forecasts made up to and including t0
    available_init_times = ds.init_time_utc.sel(
        init_time_utc=slice(None, t0_available)
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
        xr_sel = ds.sel(step=step_indexer, init_time_utc=init_time_indexer)

    else:
        # First minimise the size of the dataset we are diffing
        # - find the init times we are slicing from
        unique_init_times = np.unique(selected_init_times)
        # - find the min and max steps we slice over. Max is extended due to diff
        min_step = min(steps)
        max_step = max(steps) + sample_period_duration

        xr_min = ds.sel(
            {
                "init_time_utc": unique_init_times,
                "step": slice(min_step, max_step),
            }
        )

        # Slice out the data which does not need to be diffed
        xr_non_accum = xr_min.sel({channel_dim_name: non_accum_channels})
        xr_sel_non_accum = xr_non_accum.sel(
            step=step_indexer, init_time_utc=init_time_indexer
        )

        # Slice out the channels which need to be diffed
        xr_accum = xr_min.sel({channel_dim_name: accum_channels})

        # Take the diff and slice requested data
        xr_accum = xr_accum.diff(dim="step", label="lower")
        xr_sel_accum = xr_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

        # Join diffed and non-diffed variables
        xr_sel = xr.concat([xr_sel_non_accum, xr_sel_accum], dim=channel_dim_name)

        # Reorder the variable back to the original order
        xr_sel = xr_sel.sel({channel_dim_name: ds[channel_dim_name].values})

        # Rename the diffed channels
        xr_sel[channel_dim_name] = [
            f"diff_{v}" if v in accum_channels else v
            for v in xr_sel[channel_dim_name].values
        ]

    return xr_sel