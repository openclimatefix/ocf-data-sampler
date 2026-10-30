"""Select a time slice from a Dataset or DataArray."""

import numpy as np

from ocf_data_sampler.common.indexing import get_indices_in_sorted_unique
from ocf_data_sampler.common.time_utils import date_range, datetime_floor
from ocf_data_sampler.common.types import DataArrayLike


def select_time_slice[TArray: DataArrayLike](
    da: TArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
) -> TArray:
    """Select a time slice from a DataArray.

    The requested period is floored onto the available time grid.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
    """
    interval_bounds = datetime_floor(
        datetimes=np.array([t0 + interval_start, t0 + interval_end], dtype="datetime64[ns]"),
        freq=time_resolution,
    )
    start_ind, end_ind = get_indices_in_sorted_unique(da["time_utc"].values, interval_bounds)

    requested_times = date_range(interval_bounds[0], interval_bounds[1], freq=time_resolution)
    da_sel = da.isel(time_utc=slice(start_ind, end_ind+1))

    sliced_times = da_sel["time_utc"].values
    if not np.array_equal(sliced_times, requested_times):
        raise ValueError(
            f"Requested times ({requested_times}) do not match time steps in the sliced data "
            f"({sliced_times})"
        )

    return da_sel


def select_time_slice_nwp[TArray: DataArrayLike](
    da: TArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
    delay: np.timedelta64 | None,
) -> TArray:
    """Select a time slice from an NWP DataArray.

    The requested period is floored onto the available NWP step grid.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
        delay: Non-negative availability delay, or None for no added delay.
    """
    start_dt = t0 + interval_start
    end_dt = t0 + interval_end
    start_dt, end_dt = datetime_floor(np.array([start_dt, end_dt]), time_resolution)
    target_times = date_range(start_dt, end_dt, freq=time_resolution)

    # Unpack for convenience and so we don't need to unpack multiple times
    all_init_times = da["init_time_utc"].values
    all_steps = da["step"].values

    t0_available = t0 if delay is None else t0 - delay

    # Can't use an init-time if the start_dt is before its first step
    t0_available = min(t0_available, start_dt - all_steps[0])

    # Find the most recent available init-time <= t0_available
    selected_init_time_index = np.searchsorted(all_init_times, t0_available, side="right") - 1

    # If the selected init-time index is -1, this means that t0_available is before the first
    # available init-time in the data
    if selected_init_time_index == -1:
        raise ValueError(
            f"`t0_available` ({t0_available}) is before the first available init-time "
            f"({all_init_times[0]})"
        )

    selected_init_time = all_init_times[selected_init_time_index]

    # Find the required steps for all target-times
    required_steps = target_times - selected_init_time
    selected_step_indices = get_indices_in_sorted_unique(all_steps, required_steps)

    return da.isel(init_time_utc=selected_init_time_index, step=selected_step_indices)
