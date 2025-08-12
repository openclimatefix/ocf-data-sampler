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

    # Check that requested point lies within the data
    if not (da[x_dim].min() < x < da[x_dim].max()):
        raise ValueError(
            f"{x} is not in the interval {da[x_dim].min().values}: {da[x_dim].max().values}",
        )
    if not (da[y_dim].min() < y < da[y_dim].max()):
        raise ValueError(
            f"{y} is not in the interval {da[y_dim].min().values}: {da[y_dim].max().values}",
        )

    x_index = da.get_index(x_dim)
    y_index = da.get_index(y_dim)
    closest_x = x_index.get_indexer([x], method="nearest")[0]
    closest_y = y_index.get_indexer([y], method="nearest")[0]

    return closest_x, closest_y


def _select_padded_slice(
    da: xr.DataArray,
    left_idx: int,
    right_idx: int,
    bottom_idx: int,
    top_idx: int,
    x_dim: str,
    y_dim: str,
) -> xr.DataArray:
    """Selects spatial slice - padding where necessary if indices are out of bounds.

    Args:
        da: xarray DataArray.
        left_idx: The leftmost index of the slice.
        right_idx: The rightmost index of the slice.
        bottom_idx: The bottommost index of the slice.
        top_idx: The topmost index of the slice.
        x_dim: Name of the x dimension.
        y_dim: Name of the y dimension.

    Returns:
        An xarray DataArray with padding, if necessary.
    """
    data_width_pixels = len(da[x_dim])
    data_height_pixels = len(da[y_dim])

    left_pad_pixels = max(0, -left_idx)
    right_pad_pixels = max(0, right_idx - data_width_pixels)
    bottom_pad_pixels = max(0, -bottom_idx)
    top_pad_pixels = max(0, top_idx - data_height_pixels)

    if (left_pad_pixels > 0 and right_pad_pixels > 0) or (
        bottom_pad_pixels > 0 and top_pad_pixels > 0
    ):
        raise ValueError("Cannot pad both sides of the window")

    dx = np.median(np.diff(da[x_dim].values))
    dy = np.median(np.diff(da[y_dim].values))

    # Create a new DataArray which has indices which go outside
    # the original DataArray
    # Pad the left of the window
    if left_pad_pixels > 0:
        x_sel = np.concatenate(
            [
                da[x_dim].values[0] + np.arange(-left_pad_pixels, 0) * dx,
                da[x_dim].values[0:right_idx],
            ],
        )
        da = da.isel({x_dim: slice(0, right_idx)}).reindex({x_dim: x_sel})

    # Pad the right of the window
    elif right_pad_pixels > 0:
        x_sel = np.concatenate(
            [
                da[x_dim].values[left_idx:],
                da[x_dim].values[-1] + np.arange(1, right_pad_pixels + 1) * dx,
            ],
        )
        da = da.isel({x_dim: slice(left_idx, None)}).reindex({x_dim: x_sel})

    # No left-right padding required
    else:
        da = da.isel({x_dim: slice(left_idx, right_idx)})

    # Pad the bottom of the window
    if bottom_pad_pixels > 0:
        y_sel = np.concatenate(
            [
                da[y_dim].values[0] + np.arange(-bottom_pad_pixels, 0) * dy,
                da[y_dim].values[0:top_idx],
            ],
        )
        da = da.isel({y_dim: slice(0, top_idx)}).reindex({y_dim: y_sel})

    # Pad the top of the window
    elif top_pad_pixels > 0:
        y_sel = np.concatenate(
            [
                da[y_dim].values[bottom_idx:],
                da[y_dim].values[-1] + np.arange(1, top_pad_pixels + 1) * dy,
            ],
        )
        da = da.isel({y_dim: slice(bottom_idx, None)}).reindex({y_dim: y_sel})

    # No bottom-top padding required
    else:
        da = da.isel({y_dim: slice(bottom_idx, top_idx)})

    return da


def select_spatial_slice_pixels(
    da: xr.DataArray,
    location: Location,
    width_pixels: int,
    height_pixels: int,
    allow_partial_slice: bool = False,
) -> xr.DataArray:
    """Select spatial slice based off pixels from location point of interest.

    Args:
        da: xarray DataArray to slice from
        location: Location of interest that will be the center of the returned slice
        height_pixels: Height of the slice in pixels
        width_pixels: Width of the slice in pixels
        allow_partial_slice: Whether to allow a partial slice.

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
    pad_required = (
        left_idx < 0
        or right_idx > data_width_pixels
        or bottom_idx < 0
        or top_idx > data_height_pixels
    )

    if pad_required:
        if allow_partial_slice:
            da = _select_padded_slice(da, left_idx, right_idx, bottom_idx, top_idx, x_dim, y_dim)
        else:
            issues = []
            if left_idx < 0:
                issues.append(f"left_idx ({left_idx}) < 0")
            if right_idx > data_width_pixels:
                issues.append(f"right_idx ({right_idx}) > data_width_pixels ({data_width_pixels})")
            if bottom_idx < 0:
                issues.append(f"bottom_idx ({bottom_idx}) < 0")
            if top_idx > data_height_pixels:
                issues.append(f"top_idx ({top_idx}) > data_height_pixels ({data_height_pixels})")
            issue_details = "\n".join(issues)
            raise ValueError(
                f"Window for location {location} not available.  Padding required due to: \n"
                f"{issue_details}\n"
                "You may wish to set `allow_partial_slice=True`",
            )
    else:
        # Standard selection - without padding
        da = da.isel({x_dim: slice(left_idx, right_idx), y_dim: slice(bottom_idx, top_idx)})

    if len(da[x_dim]) != width_pixels:
        raise ValueError(f"x-dim has size {len(da[x_dim])}, expected {width_pixels}")
    if len(da[y_dim]) != height_pixels:
        raise ValueError(f"y-dim has size {len(da[y_dim])}, expected {height_pixels}")

    return da
