"""Select a time slice from a Dataset or DataArray."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import tensorstore as ts
import xarray as xr
from numpy.typing import NDArray
from xarray_tensorstore import _TensorStoreAdapter

from ocf_data_sampler.common.indexing import get_indices_in_sorted_unique
from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.time_utils import date_range, datetime_floor
from ocf_data_sampler.common.types import DataArrayLike
from ocf_data_sampler.common.xr_tensorstore import get_backing_array


def _append_nan_steps(
    data: NDArray[Any] | ts.TensorStore,
    axis: int,
    n_steps: int,
) -> NDArray[Any] | ts.TensorStore:
    """Append NaN steps along an axis, keeping TensorStore data lazy."""
    pad_shape = list(data.shape)
    pad_shape[axis] = n_steps
    if isinstance(data, ts.TensorStore):
        nan_array = ts.array(np.full(pad_shape, np.nan, dtype=data.dtype.numpy_dtype))
        return ts.concat([data, nan_array], axis=axis)
    else:
        nan_array = np.full(pad_shape, np.nan, dtype=data.dtype)
        return np.concatenate([data, nan_array], axis=axis)


def _append_lightarray_nan_times(
    da: LightDataArray,
    times: NDArray[np.datetime64],
) -> LightDataArray:
    """LightDataArray implementation of `_append_nan_times`."""
    time_axis = da.dims.index("time_utc")
    all_times = np.concatenate([da["time_utc"].values, times])
    padded_data = _append_nan_steps(da.data, time_axis, len(times))
    return LightDataArray(
        data=padded_data,
        coords={**da.coords, "time_utc": (("time_utc",), all_times)},
        dims=da.dims,
        attrs=dict(da.attrs),
        skip_validation=True,
    )


def _append_xarray_nan_times(
    da: xr.DataArray,
    times: NDArray[np.datetime64],
) -> xr.DataArray:
    """Xarray implementation of `_append_nan_times`."""
    time_axis = da.dims.index("time_utc")
    all_times = np.concatenate([da["time_utc"].values, times])
    padded_data = _append_nan_steps(get_backing_array(da), time_axis, len(times))
    # If the backing array is a TensorStore, wrap it in the internal xarray-tensorStore adapter
    if isinstance(padded_data, ts.TensorStore):
        padded_data = _TensorStoreAdapter(padded_data)
    return xr.DataArray(
        padded_data,
        coords={**da.coords.drop_vars("time_utc"), "time_utc": all_times},
        dims=da.dims,
        attrs=da.attrs,
        name=da.name,
    )


def _append_nan_times[TArray: DataArrayLike](
    da: TArray,
    times: NDArray[np.datetime64],
) -> TArray:
    """Append NaN steps at the given times along 'time_utc'.

    The times must follow on from the existing times. TensorStore-backed data stays lazy.

    Args:
        da: DataArray-like with a floating dtype and a 'time_utc' dimension. No other coordinate
            may span 'time_utc'.
        times: Times to append along 'time_utc'.

    Returns:
        A new array of the same type with the extended 'time_utc' coordinate. Other coordinates
        are shared with the input.
    """
    if isinstance(da, LightDataArray):
        return cast("TArray", _append_lightarray_nan_times(da, times))
    return cast("TArray", _append_xarray_nan_times(cast("xr.DataArray", da), times))


def select_time_slice[TArray: DataArrayLike](
    da: TArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
    candidate_delays: Sequence[np.timedelta64 | None],
) -> TArray:
    """Select a time slice, applying the first candidate delay consistent with the data.

    The requested period is floored onto the time grid. Times after `t0 - delay` are filled with
    NaN rather than read, so the result always spans the full period. A delay is consistent with
    the data when every requested time up to its cutoff is available.

    Args:
        da: The DataArray to slice from. Must have a floating dtype if any times are dropped out.
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
        candidate_delays: Non-negative availability delays in order of preference. None means no
            delay.

    Raises:
        ValueError: If no delay is consistent with the data.
    """
    interval_bounds = datetime_floor(
        datetimes=np.array([t0 + interval_start, t0 + interval_end], dtype="datetime64[ns]"),
        freq=time_resolution,
    )
    requested_times = date_range(interval_bounds[0], interval_bounds[1], freq=time_resolution)
    available_times = da["time_utc"].values

    # Find the index range of the available times that overlap with the requested interval
    start_ind = np.searchsorted(available_times, interval_bounds[0], side="left")
    end_ind = np.searchsorted(available_times, interval_bounds[1], side="right")

    requested_is_available = np.isin(requested_times, available_times[start_ind:end_ind])

    # Number of requested times available as an unbroken run from the window start
    n_available = int(np.cumprod(requested_is_available).sum())
    n_requested = len(requested_times)

    # Use the first delay in the input list that is consistent with the available data
    for delay in candidate_delays:
        if delay is None:
            n_keep = n_requested
        else:
            n_keep = int(np.searchsorted(requested_times, t0 - delay, side="right"))
        if n_keep <= n_available:
            break
    else:
        raise ValueError(
            f"No delay in {candidate_delays} is consistent with the available data. Requested "
            f"times {requested_times} have availability {requested_is_available}"
        )

    # The first n_keep requested times are available as an unbroken run from start_ind. Later
    # times are dropped out and replaced with NaN
    da_sel = da.isel(time_utc=slice(start_ind, start_ind + n_keep))
    if n_keep < n_requested:
        da_sel = _append_nan_times(da_sel, requested_times[n_keep:])

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
