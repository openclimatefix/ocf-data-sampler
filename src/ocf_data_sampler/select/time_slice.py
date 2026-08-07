"""Select a time slice from a Dataset or DataArray."""

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import get_indices_in_sorted_unique
from ocf_data_sampler.common.time_utils import date_range, datetime_ceil
from ocf_data_sampler.common.types import TArray


def select_time_slice(
    da: TArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
) -> TArray:
    """Select a time slice from a DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
    """
    interval_bounds = np.array([t0 + interval_start, t0 + interval_end], dtype="datetime64[ns]")
    ceil_interval_bounds = datetime_ceil(interval_bounds, time_resolution)
    start_ind, end_ind = get_indices_in_sorted_unique(da["time_utc"].values, ceil_interval_bounds)

    expected_n_timesteps = int((interval_end - interval_start) // time_resolution) + 1
    selected_n_timesteps = end_ind - start_ind + 1

    if selected_n_timesteps != expected_n_timesteps:
        raise ValueError(
            f"Requested interval ({interval_start} to {interval_end}) does not match "
            f"the number of time steps in the sliced data ({selected_n_timesteps}); "
            f"expected {expected_n_timesteps}"
        )

    return da.isel(time_utc=slice(start_ind, end_ind+1))


def select_time_slice_nwp(
    da: TArray,
    t0: np.datetime64,
    interval_start: np.timedelta64,
    interval_end: np.timedelta64,
    time_resolution: np.timedelta64,
    dropout_timedeltas: NDArray[np.timedelta64] | None,
    dropout_frac: float | list[float],
) -> TArray:
    """Select a time slice from an NWP DataArray.

    Args:
        da: The DataArray to slice from
        t0: The init-time
        interval_start: The start of the interval with respect to t0
        interval_end: The end of the interval with respect to t0
        time_resolution: Distance between neighbouring timestamps
        dropout_timedeltas: List of possible timedeltas before t0 where data availability may start
        dropout_frac: Either a float dropout probability or a list of per-timedelta
            probabilities. For list inputs, values must be in [0, 1], sum to <= 1,
            and match `dropout_timedeltas` length.
    """
    start_dt = t0 + interval_start
    end_dt = t0 + interval_end
    start_dt, end_dt = datetime_ceil(np.array([start_dt, end_dt]), time_resolution)
    target_times = date_range(start_dt, end_dt, freq=time_resolution)

    # Unpack for convenience and so we don't need to unpack multiple times
    all_init_times = da["init_time_utc"].values
    all_steps = da["step"].values

    t0_available = _get_nwp_dropout_available_time(t0, dropout_timedeltas, dropout_frac)

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


def _get_nwp_dropout_available_time(
    t0: np.datetime64,
    dropout_timedeltas: NDArray[np.timedelta64] | None,
    dropout_frac: float | list[float],
) -> np.datetime64:
    """Choose the available-time timestamp after applying configured dropout."""
    if dropout_timedeltas is None or len(dropout_timedeltas) == 0 or dropout_frac == 0:
        return t0

    if np.any(dropout_timedeltas > np.timedelta64(0, "ns")):
        raise ValueError("`dropout_timedeltas` must be negative or zero")

    if isinstance(dropout_frac, float | int):
        dropout_sum = dropout_frac
        dropout_probs = [dropout_frac / len(dropout_timedeltas)] * len(dropout_timedeltas)

    else:
        dropout_sum = sum(dropout_frac)
        dropout_probs = [*dropout_frac]

    if dropout_sum == 0:
        return t0

    if not 0 <= dropout_sum <= 1:
        raise ValueError(f"The sum of `dropout_frac` ({dropout_frac}) must be in range [0, 1]")
    if not all(0 <= p <= 1 for p in dropout_probs):
        raise ValueError(f"All `dropout_frac` ({dropout_frac}) values must be in range [0, 1]")
    if len(dropout_timedeltas) != len(dropout_probs):
        raise ValueError(
            "`dropout_timedeltas` and `dropout_frac` must have the same length or `dropout_frac` "
            "must be a float"
        )

    dropout_choices: list[np.timedelta64 | None] = [*dropout_timedeltas]

    # Add a None option to represent no dropout, with probability 1 - sum(dropout_frac)
    dropout_choices.append(None)
    dropout_probs.append(1 - dropout_sum)

    selected_dropout = np.random.choice(dropout_choices, p=dropout_probs)
    if selected_dropout is None:
        return t0
    return t0 + selected_dropout
