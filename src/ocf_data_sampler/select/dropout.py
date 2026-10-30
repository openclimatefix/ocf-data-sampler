"""Functions for randomly delaying availability and masking sequential data."""

from collections.abc import Mapping

import numpy as np

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.common.types import DataArrayLike


def choose_delay(delay_minute_probabilities: Mapping[int, float]) -> np.timedelta64 | None:
    """Draw a non-negative availability delay from validated minute/probability pairs.

    Unassigned probability returns None, meaning no additional availability cutoff.
    Iterating the mapping avoids allocating choice and probability arrays for every sample.

    Args:
        delay_minute_probabilities: Non-empty mapping of non-negative integer delays in
            minutes to positive probabilities summing to at most 1. Inputs must already
            be validated.

    Returns:
        The chosen delay, or None for the remaining probability of no added delay.

    """
    draw = np.random.random()
    cumulative_probability = 0.0
    for delay_minutes, probability in delay_minute_probabilities.items():
        cumulative_probability += probability
        if draw < cumulative_probability:
            return minutes(delay_minutes)
    return None


def apply_dropout(
    da: DataArrayLike,
    t0: np.datetime64,
    dropout_delay: np.timedelta64 | None,
) -> None:
    """Replace observations strictly after t0 minus the selected delay with NaNs in-place.

    Requires materialised NumPy data. A None delay leaves observations unchanged;
    a zero delay masks only observations after t0. Filling happens later in preprocessing.

    Args:
        da: DataArray-like with a 'time_utc' coordinate.
        t0: Forecast reference time.
        dropout_delay: Non-negative availability delay, or None for no dropout.
    """
    if dropout_delay is None:
        return
    if not isinstance(da.data, np.ndarray):
        raise ValueError(
            f"Dropout can only be applied to DataArrays with numpy data. Got: {type(da.data)}"
        )

    times = da["time_utc"].values
    keep = times <= t0 - dropout_delay
    axis = da.dims.index("time_utc")
    mask = np.expand_dims(keep, tuple(i for i in range(da.data.ndim) if i != axis))
    da.data = np.where(mask, da.data, np.asarray(np.nan, dtype=da.data.dtype))
