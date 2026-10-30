"""Select spatial slices."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.types import TArray


def _get_central_index(
    values: NDArray[np.number[Any]],
    val: float,
    method: Literal["nearest", "left"]
) -> int:
    """Find pixel index location closest to given value.

    This function assumes `values` are strictly increasing

    Args:
        values: The array of values to search.
        val: The value to find the closest index for.
        method: Method to use for finding the index ("nearest" or "left"). If set to "nearest", the
            index of the closest value will be returned. If set to "left", the index of the closest
            value that is less than or equal to `val` will be returned.

    Returns:
        The index of the closest value.

    Raises:
        ValueError: If the value is outside the bounds of the array.
    """
    # Check that requested point lies within the data
    if not (values[0] < val < values[-1]):
        raise ValueError(f"{val} is not in the interval {values[0]}: {values[-1]}")

    if method == "left":
        # Get the index of the closest value that is less than or equal to val
        index = np.searchsorted(values, val, side="right") - 1
    elif method == "nearest":
        # Get the index of the closest value to val
        index = np.searchsorted(values, val, side="left") - 1
        if index < len(values) - 1 and (abs(values[index+1] - val) < abs(values[index] - val)):
            index += 1
    else:
        raise ValueError(f"Unknown method: {method}")

    return int(index)


def _get_window_bounds(central_index: int, window_size: int) -> tuple[int, int]:
    """Get the lower and upper bounds of a window around a central index."""
    low_pad = (window_size+1)//2 - 1
    high_pad = window_size//2 + 1

    low_idx = int(central_index - low_pad)
    high_idx = int(central_index + high_pad)

    return low_idx, high_idx


def _validate_window_slice(
    window_slice: tuple[int, int, int, int],
    total_size: tuple[int, int],
) -> None:
    """Validate that the window slice is within the bounds of the data."""
    left_idx, right_idx, bottom_idx, top_idx = window_slice
    total_width, total_height = total_size

    if left_idx < 0:
        raise ValueError(f"Left index ({left_idx}) < 0")
    if right_idx > total_width:
        raise ValueError(f"Right index ({right_idx}) > total_width ({total_width})")
    if bottom_idx < 0:
        raise ValueError(f"Bottom index ({bottom_idx}) < 0")
    if top_idx > total_height:
        raise ValueError(f"Top index ({top_idx}) > total_height ({total_height})")


def select_spatial_slice(
    da: TArray,
    *,
    x: float,
    y: float,
    x_dim: str,
    y_dim: str,
    width_pixels: int,
    height_pixels: int,
) -> TArray:
    """Select a spatial window of the requested pixel size around the given centre.

    Args:
        da: DataArray-like object to slice from
        x: X coordinate of the window centre in the source's coordinate system
        y: Y coordinate of the window centre in the source's coordinate system
        x_dim: Name of the source's X coordinate dimension
        y_dim: Name of the source's Y coordinate dimension
        height_pixels: Height of the window in pixels.
        width_pixels: Width of the window in pixels.

    Returns:
        The selected DataArray-like slice.
    """
    x_values = da[x_dim].values
    y_values = da[y_dim].values

    # If odd window size, get index of nearest pixel, else get index of closest pixel to the left
    # to ensure the location is centred within the returned slice
    x_method = "nearest" if (width_pixels % 2) == 1 else "left"
    y_method = "nearest" if (height_pixels % 2) == 1 else "left"
    x_index = _get_central_index(x_values, x, method=x_method)
    y_index = _get_central_index(y_values, y, method=y_method)

    left_idx, right_idx = _get_window_bounds(x_index, width_pixels)
    bottom_idx, top_idx = _get_window_bounds(y_index, height_pixels)

    data_width_pixels = len(x_values)
    data_height_pixels = len(y_values)

    _validate_window_slice(
        window_slice=(left_idx, right_idx, bottom_idx, top_idx),
        total_size=(data_width_pixels, data_height_pixels),
    )

    return da.isel({x_dim: slice(left_idx, right_idx), y_dim: slice(bottom_idx, top_idx)})


def select_envelope_of_spatial_windows(
    da: TArray,
    *,
    centres: list[tuple[float, float]],
    x_dim: str,
    y_dim: str,
    width_pixels: int,
    height_pixels: int,
) -> TArray:
    """Select a rectangular region enclosing the spatial windows around all given centres.

    Args:
        da: DataArray-like object to slice from
        centres: Paired (x, y) window centres in the source's coordinate system
        x_dim: Name of the source's X coordinate dimension
        y_dim: Name of the source's Y coordinate dimension
        height_pixels: Height of each window in pixels.
        width_pixels: Width of each window in pixels.

    Returns:
        The selected DataArray-like slice.
    """
    if len(centres) == 0:
        raise ValueError("`centres` is empty - there is no region to cover")

    x_values = da[x_dim].values
    y_values = da[y_dim].values

    x_method = "nearest" if (width_pixels % 2) == 1 else "left"
    y_method = "nearest" if (height_pixels % 2) == 1 else "left"

    data_width_pixels = len(x_values)
    data_height_pixels = len(y_values)

    idx_x_min: int = data_width_pixels
    idx_x_max: int = 0
    idx_y_min: int = data_height_pixels
    idx_y_max: int = 0

    for x, y in centres:
        x_index = _get_central_index(x_values, x, method=x_method)
        y_index = _get_central_index(y_values, y, method=y_method)
        idx_x_min = min(idx_x_min, x_index)
        idx_x_max = max(idx_x_max, x_index)
        idx_y_min = min(idx_y_min, y_index)
        idx_y_max = max(idx_y_max, y_index)

    left_idx, _ = _get_window_bounds(idx_x_min, width_pixels)
    _, right_idx = _get_window_bounds(idx_x_max, width_pixels)
    bottom_idx, _ = _get_window_bounds(idx_y_min, height_pixels)
    _, top_idx = _get_window_bounds(idx_y_max, height_pixels)

    _validate_window_slice(
        window_slice=(left_idx, right_idx, bottom_idx, top_idx),
        total_size=(data_width_pixels, data_height_pixels),
    )

    # We add a conditional buffer so that this crop is compatible with select_spatial_slice
    # for all of the locations. select_spatial_slice checks that the location centres are
    # strictly within the spatial bounds of the data. If using a window size of 1, the locations at
    # the edge of the crop calculated above could fail the check x_min < loc.x < x_max. There is a
    # similar edge case for window size of 2 pixels if a location at the edge of the crop sits
    # exactly on a boundary pixel i.e. x_min == loc.x
    if width_pixels <= 2:
        left_idx = max(0, left_idx - 1)
        right_idx = min(data_width_pixels, right_idx + 1)

    if height_pixels <= 2:
        bottom_idx = max(0, bottom_idx - 1)
        top_idx = min(data_height_pixels, top_idx + 1)

    return da.isel({x_dim: slice(left_idx, right_idx), y_dim: slice(bottom_idx, top_idx)})
