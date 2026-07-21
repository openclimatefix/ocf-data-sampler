"""Select spatial slices."""

from typing import Literal

import numpy as np

from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.spatial import Location, find_coord_system


def _get_central_index(
    values: np.ndarray,
    val: float,
    method: Literal["nearest", "left"],
) -> int:
    """Find pixel index location closest to given value.

    Args:
        values: The array of values to search.
        val: The value to find the closest index for.
        method: Method to use for finding the index ("nearest" or "left").

    Returns:
        The index of the closest value.

    Raises:
        ValueError: If the value is outside the bounds of the array.
    """
    # Check that requested point lies within the data
    if not (values[0] < val < values[-1]):
        raise ValueError(
            f"{val} is not in the interval {values[0]}: {values[-1]}",
        )

    def get_nearest_index(vals: np.ndarray, val: float) -> int:
        """Get the index of the nearest value in vals to val."""
        idx = np.searchsorted(vals, val, side="left") - 1
        if idx < len(vals) - 1 and (abs(vals[idx+1] - val) < abs(vals[idx] - val)):
            idx += 1
        return idx

    if method == "left":
        index = np.searchsorted(values, val, side="left") - 1
    elif method == "nearest":
        index = get_nearest_index(values, val)
    else:
        raise ValueError(f"Unknown method: {method}")

    return index


def _get_window_bounds(
    central_index: int,
    window_size: int,
) -> tuple[int, int]:
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

    slice_unavailable = (
        left_idx < 0
        or right_idx > total_width
        or bottom_idx < 0
        or top_idx > total_height
    )

    if slice_unavailable:
        issues = []
        if left_idx < 0:
            issues.append(f"left index ({left_idx}) < 0")
        if right_idx > total_width:
            issues.append(f"right index ({right_idx}) > total_width ({total_width})")
        if bottom_idx < 0:
            issues.append(f"bottom index ({bottom_idx}) < 0")
        if top_idx > total_height:
            issues.append(f"top index ({top_idx}) > total_height ({total_height})")
        issue_details = "\n - ".join(issues)
        raise ValueError(f"Slice is unavailable:\n - {issue_details}")

def select_spatial_slice_pixels(
    da: TArray,
    location: Location,
    width_pixels: int,
    height_pixels: int,
) -> TArray:
    """Select spatial slice based off pixels from location point of interest.

    Args:
        da: DataArray-like object to slice from
        location: Location of interest that will be the center of the returned slice
        height_pixels: Height of the slice in pixels
        width_pixels: Width of the slice in pixels

    Returns:
        The selected DataArray-like slice.
    """
    target_coords, x_dim, y_dim = find_coord_system(da)

    x, y = location.in_coord_system(target_coords)

    x_values = da[x_dim].values
    y_values = da[y_dim].values

    # If odd window size, get index of nearest pixel, else get index of left pixel to ensure the
    # location is within the returned slice
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


def select_spatial_slice_pixels_multiple(
    da: TArray,
    locations: list[Location],
    width_pixels: int,
    height_pixels: int,
) -> TArray:
    """Select spatial slice which covers all given locations.

    Args:
        da: DataArray-like object to slice from
        locations: List of locations of interest that will be covered by the returned slice
        height_pixels: Height of the slice in pixels
        width_pixels: Width of the slice in pixels

    Returns:
        The selected DataArray-like slice.
    """
    target_coords, x_dim, y_dim = find_coord_system(da)

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

    for location in locations:
        x, y = location.in_coord_system(target_coords)
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

    return da.isel({x_dim: slice(left_idx, right_idx), y_dim: slice(bottom_idx, top_idx)})
