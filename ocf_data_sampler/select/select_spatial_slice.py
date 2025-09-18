"""Select spatial slices."""

import numpy as np
import xarray as xr

from ocf_data_sampler.select.geospatial import find_coord_system
from ocf_data_sampler.select.location import Location


def _get_pixel_index_location(da: xr.DataArray, location: Location) -> tuple[int, int]:
    """Find pixel index location closest to given Location.

    Args:
        da: The xarray DataArray.
        location: The Location object representing the point of interest.

    Returns:
        The pixel indices.

    Raises:
        ValueError: If the location is outside the bounds of the DataArray.
    """
    target_coords, x_dim, y_dim = find_coord_system(da)

    x, y = location.in_coord_system(target_coords)

    x_vals = da[x_dim].values
    y_vals = da[y_dim].values

    # Check that requested point lies within the data
    if not (x_vals[0] < x < x_vals[-1]):
        raise ValueError(
            f"{x} is not in the interval {x_vals[0]}: {x_vals[-1]}",
        )
    if not (y_vals[0] < y < y_vals[-1]):
        raise ValueError(
            f"{y} is not in the interval {y_vals[0]}: {y_vals[-1]}",
        )

    closest_x = np.argmin(np.abs(x_vals - x))
    closest_y = np.argmin(np.abs(y_vals - y))

    return closest_x, closest_y


def select_spatial_slice_pixels(
    da: xr.DataArray,
    location: Location,
    width_pixels: int,
    height_pixels: int,
) -> xr.DataArray:
    """Select spatial slice based off pixels from location point of interest.

    Args:
        da: xarray DataArray to slice from
        location: Location of interest that will be the center of the returned slice
        height_pixels: Height of the slice in pixels
        width_pixels: Width of the slice in pixels

    Returns:
        The selected DataArray slice.

    Raises:
        ValueError: If the dimensions are not even or the slice is not allowed
                    when padding is required.
    """
    if (width_pixels % 2) != 0:
        raise ValueError("Width must be an even number")
    if (height_pixels % 2) != 0:
        raise ValueError("Height must be an even number")

    _, x_dim, y_dim = find_coord_system(da)
    center_idx_x, center_idx_y = _get_pixel_index_location(da, location)

    half_width = width_pixels // 2
    half_height = height_pixels // 2

    left_idx = int(center_idx_x - half_width)
    right_idx = int(center_idx_x + half_width)
    bottom_idx = int(center_idx_y - half_height)
    top_idx = int(center_idx_y + half_height)

    data_width_pixels = len(da[x_dim])
    data_height_pixels = len(da[y_dim])

    # Padding checks
    slice_unavailable = (
        left_idx < 0
        or right_idx > data_width_pixels
        or bottom_idx < 0
        or top_idx > data_height_pixels
    )

    if slice_unavailable:
        issues = []
        if left_idx < 0:
            issues.append(f"left_idx ({left_idx}) < 0")
        if right_idx > data_width_pixels:
            issues.append(f"right_idx ({right_idx}) > data_width_pixels ({data_width_pixels})")
        if bottom_idx < 0:
            issues.append(f"bottom_idx ({bottom_idx}) < 0")
        if top_idx > data_height_pixels:
            issues.append(f"top_idx ({top_idx}) > data_height_pixels ({data_height_pixels})")
        issue_details = "\n - ".join(issues)
        raise ValueError(f"Window for location {location} not available: \n - {issue_details}")

    # Standard selection - without padding
    da = da.isel({x_dim: slice(left_idx, right_idx), y_dim: slice(bottom_idx, top_idx)})

    return da
