"""Clipping and standardisation of NumPy arrays."""

import numpy as np
from numpy.typing import NDArray


def clip_and_standardise(
    data: NDArray[np.float32],
    *,
    clip_min: NDArray[np.float32],
    clip_max: NDArray[np.float32],
    mean: NDArray[np.float32],
    std: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Clip, subtract the mean, and divide by the standard deviation.

    Args:
        data: Input array.
        clip_min: Lower clipping bounds, broadcastable to the input shape.
        clip_max: Upper clipping bounds, broadcastable to the input shape.
        mean: Means to subtract, broadcastable to the input shape.
        std: Standard deviations to divide by, broadcastable to the input shape.

    Returns:
        A new array containing the clipped and standardised values.
    """
    return (data.clip(clip_min, clip_max) - mean) / std
