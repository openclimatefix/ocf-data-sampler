import xarray as xr
import pandas as pd
import numpy as np

def _sel_default(
        da: xr.DataArray, 
        start_dt: pd.Timestamp, 
        end_dt: pd.Timestamp, 
        sample_period_duration: pd.Timedelta,
    ) -> xr.DataArray:
    """Select a time slice from a DataArray, without filling missing times."""
    return da.sel(time_utc=slice(start_dt, end_dt))

def select_time_slice(
    ds: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    sample_period_duration: pd.Timedelta,
    fill_selection: bool = False,
    max_steps_gap: int = 0,
):
    """Select a time slice from a Dataset or DataArray."""
    assert max_steps_gap >= 0, "max_steps_gap must be >= 0 "
    
    _sel = _sel_default

    t0_datetime_utc = pd.Timestamp(t0)
    start_dt = t0_datetime_utc + interval_start
    end_dt = t0_datetime_utc + interval_end

    start_dt = start_dt.ceil(sample_period_duration)
    end_dt = end_dt.ceil(sample_period_duration)

    return _sel(ds, start_dt, end_dt, sample_period_duration)

def select_time_slice_nwp(
    da: xr.DataArray,
    t0: pd.Timestamp,
    interval_start: pd.Timedelta,
    interval_end: pd.Timedelta,
    sample_period_duration: pd.Timedelta,
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
    consider_dropout = (dropout_timedeltas is not None) and dropout_frac > 0

    accum_channels = np.intersect1d(
        da[channel_dim_name].values, accum_channels
    )
    non_accum_channels = np.setdiff1d(
        da[channel_dim_name].values, accum_channels
    )

    start_dt = (t0 + interval_start).ceil(sample_period_duration)
    end_dt = (t0 + interval_end).ceil(sample_period_duration)

    target_times = pd.date_range(start_dt, end_dt, freq=sample_period_duration)

    if consider_dropout and (np.random.uniform() < dropout_frac):
        dt = np.random.choice(dropout_timedeltas)
        t0_available = t0 + dt
    else:
        t0_available = t0

    available_init_times = da.init_time_utc.sel(
        init_time_utc=slice(None, t0_available)
    )

    selected_init_times = available_init_times.sel(
        init_time_utc=target_times,
        method="ffill",
    ).values

    steps = target_times - selected_init_times

    coords = {"target_time_utc": target_times}
    init_time_indexer = xr.DataArray(selected_init_times, coords=coords)
    step_indexer = xr.DataArray(steps, coords=coords)

    if len(accum_channels) == 0:
        da_sel = da.sel(step=step_indexer, init_time_utc=init_time_indexer)

    else:
        unique_init_times = np.unique(selected_init_times)
        min_step = min(steps)
        max_step = max(steps) + sample_period_duration

        da_min = da.sel(
            {
                "init_time_utc": unique_init_times,
                "step": slice(min_step, max_step),
            }
        )

        da_non_accum = da_min.sel({channel_dim_name: non_accum_channels})
        da_sel_non_accum = da_non_accum.sel(
            step=step_indexer, init_time_utc=init_time_indexer
        )

        da_accum = da_min.sel({channel_dim_name: accum_channels})
        da_accum = da_accum.diff(dim="step", label="lower")
        da_sel_accum = da_accum.sel(step=step_indexer, init_time_utc=init_time_indexer)

        da_sel = xr.concat([da_sel_non_accum, da_sel_accum], dim=channel_dim_name)
        da_sel = da_sel.sel({channel_dim_name: da[channel_dim_name].values})

        da_sel[channel_dim_name] = [
            f"diff_{v}" if v in accum_channels else v
            for v in da_sel[channel_dim_name].values
        ]

    return da_sel
