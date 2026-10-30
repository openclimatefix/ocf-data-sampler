"""Clipping, standardisation, and generation capacity normalisation."""

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import get_string_index
from ocf_data_sampler.common.types import DataArrayLike


def clip_and_standardise(
    da: DataArrayLike,
    *,
    clip_min: NDArray[np.float32],
    clip_max: NDArray[np.float32],
    mean: NDArray[np.float32],
    std: NDArray[np.float32],
) -> None:
    """Clip, subtract the mean, and divide by the standard deviation.

    Replaces `.data` on the input DataArray without modifying shared source values.

    Args:
        da: Input DataArray-like.
        clip_min: Lower clipping bounds, broadcastable to the input shape.
        clip_max: Upper clipping bounds, broadcastable to the input shape.
        mean: Means to subtract, broadcastable to the input shape.
        std: Standard deviations to divide by, broadcastable to the input shape.
    """
    da.data = (da.data.clip(clip_min, clip_max) - mean) / std


def normalise_generation_by_capacity(da: DataArrayLike) -> None:
    """Rescale `generation_mw` to a capacity factor, leaving `capacity_mwp` unchanged.

    Replaces `.data` on the input DataArray without modifying shared source values.

    Where capacity is zero, the normalised generation value is zero.

    Args:
        da: Generation DataArray-like with a `gen_param` dimension
    """
    gen_params = da["gen_param"].values
    gen_idx = get_string_index(gen_params, "generation_mw")
    cap_idx = get_string_index(gen_params, "capacity_mwp")

    generation_values = da.isel(gen_param=gen_idx).values
    capacity_values = da.isel(gen_param=cap_idx).values

    normalised = np.divide(
        generation_values,
        capacity_values,
        out=np.zeros_like(generation_values),
        where=capacity_values != 0,
    )

    new_data = da.data.copy()
    index = [slice(None)] * new_data.ndim
    index[da.dims.index("gen_param")] = gen_idx
    new_data[tuple(index)] = normalised
    da.data = new_data
