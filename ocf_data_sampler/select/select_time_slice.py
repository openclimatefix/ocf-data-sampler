"""Select a time slice from a Dataset or DataArray."""

import numpy as np
import xarray as xr

from ocf_data_sampler.select import get_indices_in_sorted_unique
from ocf_data_sampler.time_utils import date_range, datetime_ceil


def select_time_slice(
    da: xr.DataArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
) -> xr.DataArray:
    """Select a time slice from a DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
    """
    date_range = np.array([t0 + interval_start, t0 + interval_end])
    ceil_date_range = datetime_ceil(date_range, time_resolution)
    start_ind, end_ind = get_indices_in_sorted_unique(da.time_utc.values, ceil_date_range)

    return da.isel(time_utc=slice(start_ind, end_ind+1))


def select_time_slice_nwp(
    da: xr.DataArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
    dropout_timedeltas: list[np.timedelta64] | None = None,
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
        if not all(t < np.timedelta64(0) for t in dropout_timedeltas):
            raise ValueError("dropout timedeltas must be negative")
        if len(dropout_timedeltas) < 1:
            raise ValueError("dropout timedeltas must have at least one element")

    if not (0 <= dropout_frac <= 1):
        raise ValueError("dropout_frac must be between 0 and 1")

    consider_dropout = len(dropout_timedeltas) > 0 and dropout_frac > 0

    start_dt = t0 + interval_start
    end_dt = t0 + interval_end
    start_dt, end_dt = datetime_ceil(np.array([start_dt, end_dt]), time_resolution)
    target_times = date_range(start_dt, end_dt, freq=time_resolution)

    # Unpack for convenience and so we don't need to unpack multiple times
    all_init_times = da.init_time_utc.values
    all_steps = da.step.values

    # Potentially apply NWP dropout
    if consider_dropout and (np.random.uniform() < dropout_frac):
        t0_available = t0 + np.random.choice(dropout_timedeltas)
    else:
        t0_available = t0

    # Can't use an init-time if the start_dt is before its first step
    t0_available = min(t0_available, start_dt - all_steps[0])

    # Find the most recent available init-time <= t0_available
    selected_init_time_index = np.searchsorted(all_init_times, t0_available, side="right") - 1
    selected_init_time = all_init_times[selected_init_time_index]

    # Find the required steps for all target-times
    required_steps = target_times - selected_init_time
    selected_step_indices = get_indices_in_sorted_unique(all_steps, required_steps)

    return da.isel(init_time_utc=selected_init_time_index, step=selected_step_indices)
