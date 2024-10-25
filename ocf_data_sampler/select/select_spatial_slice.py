"""Select spatial slices"""

import logging

import numpy as np
import xarray as xr

from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.geospatial import (
    lon_lat_to_osgb,
    osgb_to_geostationary_area_coords,
    osgb_to_lon_lat,
    spatial_coord_type,
)


logger = logging.getLogger(__name__)


# -------------------------------- utility functions --------------------------------


def convert_coords_to_match_xarray(
        x: float | np.ndarray, 
        y: float | np.ndarray, 
        from_coords: str, 
        da: xr.DataArray
    ):
    """Convert x and y coords to cooridnate system matching xarray data

    Args:
        x: Float or array-like
        y: Float or array-like
        from_coords: String describing coordinate system of x and y
        da: DataArray to which coordinates should be matched
    """

    target_coords, *_ = spatial_coord_type(da)

    assert from_coords in ["osgb", "lon_lat"]
    assert target_coords in ["geostationary", "osgb", "lon_lat"]

    if target_coords == "geostationary":
        if from_coords == "osgb":
            x, y = osgb_to_geostationary_area_coords(x, y, da)

    elif target_coords == "lon_lat":
        if from_coords == "osgb":
            x, y = osgb_to_lon_lat(x, y)

        # else the from_coords=="lon_lat" and we don't need to convert

    elif target_coords == "osgb":
        if from_coords == "lon_lat":
            x, y = lon_lat_to_osgb(x, y)

        # else the from_coords=="osgb" and we don't need to convert

    return x, y

# TODO: This function and _get_idx_of_pixel_closest_to_poi_geostationary() should not be separate
# We should combine them, and consider making a Coord class to help with this
def _get_idx_of_pixel_closest_to_poi(
    da: xr.DataArray,
    location: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        da: xarray DataArray
        location: Location to find index of
    Returns:
        The Location for the center pixel
    """
    xr_coords, x_dim, y_dim = spatial_coord_type(da)

    if xr_coords not in ["osgb", "lon_lat"]:
        raise NotImplementedError(f"Only 'osgb' and 'lon_lat' are supported - not '{xr_coords}'")

    # Convert location coords to match xarray data
    x, y = convert_coords_to_match_xarray(
        location.x,
        location.y,
        from_coords=location.coordinate_system,
        da=da,
    )

    # Check that the requested point lies within the data
    assert da[x_dim].min() < x < da[x_dim].max()
    assert da[y_dim].min() < y < da[y_dim].max()

    x_index = da.get_index(x_dim)
    y_index = da.get_index(y_dim)

    closest_x = x_index.get_indexer([x], method="nearest")[0]
    closest_y = y_index.get_indexer([y], method="nearest")[0]

    return Location(x=closest_x, y=closest_y, coordinate_system="idx")


def _get_idx_of_pixel_closest_to_poi_geostationary(
    da: xr.DataArray,
    center_osgb: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        da: xarray DataArray
        center_osgb: Center in OSGB coordinates

    Returns:
        Location for the center pixel in geostationary coordinates
    """

    _, x_dim, y_dim = spatial_coord_type(da)

    x, y = osgb_to_geostationary_area_coords(x=center_osgb.x, y=center_osgb.y, xr_data=da)
    center_geostationary = Location(x=x, y=y, coordinate_system="geostationary")

    # Check that the requested point lies within the data
    assert da[x_dim].min() < x < da[x_dim].max(), \
        f"{x} is not in the interval {da[x_dim].min().values}: {da[x_dim].max().values}"
    assert da[y_dim].min() < y < da[y_dim].max(), \
        f"{y} is not in the interval {da[y_dim].min().values}: {da[y_dim].max().values}"

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = np.searchsorted(da[x_dim].values, center_geostationary.x)
    y_index_at_center = np.searchsorted(da[y_dim].values, center_geostationary.y)

    return Location(x=x_index_at_center, y=y_index_at_center, coordinate_system="idx")


# ---------------------------- sub-functions for slicing ----------------------------


def _select_partial_spatial_slice_pixels(
    da,
    left_idx,
    right_idx,
    bottom_idx,
    top_idx,
    left_pad_pixels,
    right_pad_pixels,
    bottom_pad_pixels,
    top_pad_pixels,
    x_dim,
    y_dim,
):
    """Return spatial window of given pixel size when window partially overlaps input data"""

    # We should never be padding on both sides of a window. This would mean our desired window is 
    # larger than the size of the input data
    assert left_pad_pixels==0 or right_pad_pixels==0
    assert bottom_pad_pixels==0 or top_pad_pixels==0

    dx = np.median(np.diff(da[x_dim].values))
    dy = np.median(np.diff(da[y_dim].values))

    # Pad the left of the window
    if left_pad_pixels > 0:
        x_sel = np.concatenate(
            [
                da[x_dim].values[0] + np.arange(-left_pad_pixels, 0) * dx,
                da[x_dim].values[0:right_idx],
            ]
        )
        da = da.isel({x_dim: slice(0, right_idx)}).reindex({x_dim: x_sel})

    # Pad the right of the window
    elif right_pad_pixels > 0:
        x_sel = np.concatenate(
            [
                da[x_dim].values[left_idx:],
                da[x_dim].values[-1] + np.arange(1, right_pad_pixels + 1) * dx,
            ]
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
            ]
        )
        da = da.isel({y_dim: slice(0, top_idx)}).reindex({y_dim: y_sel})

    # Pad the top of the window
    elif top_pad_pixels > 0:
        y_sel = np.concatenate(
            [
                da[y_dim].values[bottom_idx:],
                da[y_dim].values[-1] + np.arange(1, top_pad_pixels + 1) * dy,
            ]
        )
        da = da.isel({y_dim: slice(left_idx, None)}).reindex({y_dim: y_sel})

    # No bottom-top padding required
    else:
        da = da.isel({y_dim: slice(bottom_idx, top_idx)})

    return da


def _select_spatial_slice_pixels(
    da: xr.DataArray, 
    center_idx: Location, 
    width_pixels: int, 
    height_pixels: int, 
    x_dim: str, 
    y_dim: str, 
    allow_partial_slice: bool,
):
    """Select a spatial slice from an xarray object

    Args:
        da: xarray DataArray to slice from
        center_idx: Location object describing the centre of the window with index coordinates
        width_pixels: Window with in pixels
        height_pixels: Window height in pixels
        x_dim: Name of the x-dimension in `da`
        y_dim: Name of the y-dimension in `da`
        allow_partial_slice: Whether to allow a partially filled window
    """

    assert center_idx.coordinate_system == "idx"
    # TODO: It shouldn't take much effort to allow height and width to be odd
    assert (width_pixels % 2)==0, "Width must be an even number"
    assert (height_pixels % 2)==0, "Height must be an even number"

    half_width = width_pixels // 2
    half_height = height_pixels // 2

    left_idx = int(center_idx.x - half_width)
    right_idx = int(center_idx.x + half_width)
    bottom_idx = int(center_idx.y - half_height)
    top_idx = int(center_idx.y + half_height)

    data_width_pixels = len(da[x_dim])
    data_height_pixels = len(da[y_dim])

    left_pad_required = left_idx < 0
    right_pad_required = right_idx > data_width_pixels
    bottom_pad_required = bottom_idx < 0
    top_pad_required = top_idx > data_height_pixels

    pad_required = left_pad_required | right_pad_required | bottom_pad_required | top_pad_required

    if pad_required:
        if allow_partial_slice:

            left_pad_pixels = (-left_idx) if left_pad_required else 0
            right_pad_pixels = (right_idx - data_width_pixels) if right_pad_required else 0

            bottom_pad_pixels = (-bottom_idx) if bottom_pad_required else 0
            top_pad_pixels = (top_idx - data_height_pixels) if top_pad_required else 0


            da = _select_partial_spatial_slice_pixels(
                da,
                left_idx,
                right_idx,
                bottom_idx,
                top_idx,
                left_pad_pixels,
                right_pad_pixels,
                bottom_pad_pixels,
                top_pad_pixels,
                x_dim,
                y_dim,
            )
        else:
            raise ValueError(
                f"Window for location {center_idx} not available. Missing (left, right, bottom, "
                f"top) pixels  = ({left_pad_required}, {right_pad_required}, "
                f"{bottom_pad_required}, {top_pad_required}). "
                f"You may wish to set `allow_partial_slice=True`"
            )

    else:
        da = da.isel(
            {
                x_dim: slice(left_idx, right_idx),
                y_dim: slice(bottom_idx, top_idx),
            }
        )

    assert len(da[x_dim]) == width_pixels, (
        f"Expected x-dim len {width_pixels} got {len(da[x_dim])} "
        f"for location {center_idx} for slice {left_idx}:{right_idx}"
    )
    assert len(da[y_dim]) == height_pixels, (
        f"Expected y-dim len {height_pixels} got {len(da[y_dim])} "
        f"for location {center_idx} for slice {bottom_idx}:{top_idx}"
    )

    return da


# ---------------------------- main functions for slicing ---------------------------


def select_spatial_slice_pixels(
    da: xr.DataArray,
    location: Location,
    width_pixels: int,
    height_pixels: int,
    allow_partial_slice: bool = False,
):
    """
    Select spatial slice based off pixels from location point of interest

    If `allow_partial_slice` is set to True, then slices may be made which intersect the border
    of the input data. The additional x and y cordinates that would be required for this slice
    are extrapolated based on the average spacing of these coordinates in the input data.
    However, currently slices cannot be made where the centre of the window is outside of the
    input data.

    Args:
        da: xarray DataArray to slice from
        location: Location of interest
        height_pixels: Height of the slice in pixels
        width_pixels: Width of the slice in pixels
        allow_partial_slice: Whether to allow a partial slice.
    """

    xr_coords, x_dim, y_dim = spatial_coord_type(da)

    if xr_coords == "geostationary":
        center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(da, location)
    else:
        center_idx: Location = _get_idx_of_pixel_closest_to_poi(da, location)

    selected = _select_spatial_slice_pixels(
        da,
        center_idx,
        width_pixels,
        height_pixels,
        x_dim,
        y_dim,
        allow_partial_slice=allow_partial_slice,
    )

    return selected