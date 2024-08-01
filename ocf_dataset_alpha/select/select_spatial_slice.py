"""Select spatial slices"""

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

from ocf_datapipes.utils import Location
from ocf_datapipes.utils.geospatial import (
    lon_lat_to_geostationary_area_coords,
    lon_lat_to_osgb,
    osgb_to_geostationary_area_coords,
    osgb_to_lon_lat,
    spatial_coord_type,
)
from ocf_datapipes.utils.utils import searchsorted

logger = logging.getLogger(__name__)


# -------------------------------- utility functions --------------------------------


def convert_coords_to_match_xarray(x, y, from_coords, xr_data):
    """Convert x and y coords to cooridnate system matching xarray data

    Args:
        x: Float or array-like
        y: Float or array-like
        from_coords: String describing coordinate system of x and y
        xr_data: xarray data object to which coordinates should be matched
    """

    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    assert from_coords in ["osgb", "lon_lat"]
    assert xr_coords in ["geostationary", "osgb", "lon_lat"]

    if xr_coords == "geostationary":
        if from_coords == "osgb":
            x, y = osgb_to_geostationary_area_coords(x, y, xr_data)

        elif from_coords == "lon_lat":
            x, y = lon_lat_to_geostationary_area_coords(x, y, xr_data)

    elif xr_coords == "lon_lat":
        if from_coords == "osgb":
            x, y = osgb_to_lon_lat(x, y)

        # else the from_coords=="lon_lat" and we don't need to convert

    elif xr_coords == "osgb":
        if from_coords == "lon_lat":
            x, y = lon_lat_to_osgb(x, y)

        # else the from_coords=="osgb" and we don't need to convert

    return x, y


def _get_idx_of_pixel_closest_to_poi(
    xr_data: xr.DataArray,
    location: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        location: Center
    Returns:
        The Location for the center pixel
    """
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    if xr_coords not in ["osgb", "lon_lat"]:
        raise NotImplementedError(f"Only 'osgb' and 'lon_lat' are supported - not '{xr_coords}'")

    # Convert location coords to match xarray data
    x, y = convert_coords_to_match_xarray(
        location.x,
        location.y,
        from_coords=location.coordinate_system,
        xr_data=xr_data,
    )

    # Check that the requested point lies within the data
    assert xr_data[xr_x_dim].min() < x < xr_data[xr_x_dim].max()
    assert xr_data[xr_y_dim].min() < y < xr_data[xr_y_dim].max()

    x_index = xr_data.get_index(xr_x_dim)
    y_index = xr_data.get_index(xr_y_dim)

    closest_x = x_index.get_indexer([x], method="nearest")[0]
    closest_y = y_index.get_indexer([y], method="nearest")[0]

    return Location(x=closest_x, y=closest_y, coordinate_system="idx")


def _get_idx_of_pixel_closest_to_poi_geostationary(
    xr_data: xr.DataArray,
    center_osgb: Location,
) -> Location:
    """
    Return x and y index location of pixel at center of region of interest.

    Args:
        xr_data: Xarray dataset
        center_osgb: Center in OSGB coordinates

    Returns:
        Location for the center pixel in geostationary coordinates
    """

    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)

    x, y = osgb_to_geostationary_area_coords(x=center_osgb.x, y=center_osgb.y, xr_data=xr_data)
    center_geostationary = Location(x=x, y=y, coordinate_system="geostationary")

    # Check that the requested point lies within the data
    assert xr_data[xr_x_dim].min() < x < xr_data[xr_x_dim].max()
    assert xr_data[xr_y_dim].min() < y < xr_data[xr_y_dim].max()

    # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
    x_index_at_center = searchsorted(
        xr_data[xr_x_dim].values, center_geostationary.x, assume_ascending=True
    )

    # y_geostationary is in descending order:
    y_index_at_center = searchsorted(
        xr_data[xr_y_dim].values, center_geostationary.y, assume_ascending=False
    )

    return Location(x=x_index_at_center, y=y_index_at_center, coordinate_system="idx")


def _get_points_from_unstructured_grids(
    xr_data: xr.DataArray,
    location: Location,
    location_idx_name: str = "values",
    num_points: int = 1,
):
    """
    Get the closest points from an unstructured grid (i.e. Icosahedral grid)

    This is primarily used for the Icosahedral grid, which is not a regular grid,
     and so is not an image

    Args:
        xr_data: Xarray dataset
        location: Location of center point
        location_idx_name: Name of the index values dimension
            (i.e. where we index into to get the lat/lon for that point)
        num_points: Number of points to return (should be width * height)

    Returns:
        The closest points from the grid
    """
    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    assert xr_coords == "lon_lat"

    # Check if need to convert from different coordinate system to lat/lon
    if location.coordinate_system == "osgb":
        longitude, latitude = osgb_to_lon_lat(x=location.x, y=location.y)
        location = Location(
            x=longitude,
            y=latitude,
            coordinate_system="lon_lat",
        )
    elif location.coordinate_system == "geostationary":
        raise NotImplementedError(
            "Does not currently support geostationary coordinates when using unstructured grids"
        )

    # Extract lat, lon, and locidx data
    lat = xr_data.longitude.values
    lon = xr_data.latitude.values
    locidx = xr_data[location_idx_name].values

    # Create a KDTree
    tree = KDTree(list(zip(lat, lon)))

    # Query with the [longitude, latitude] of your point
    _, idx = tree.query([location.x, location.y], k=num_points)

    # Retrieve the location_idxs for these grid points
    location_idxs = locidx[idx]

    data = xr_data.sel({location_idx_name: location_idxs})
    return data


# ---------------------------- sub-functions for slicing ----------------------------


def _slice_patial_spatial_pixel_window_from_xarray(
    xr_data,
    left_idx,
    right_idx,
    top_idx,
    bottom_idx,
    left_pad_pixels,
    right_pad_pixels,
    top_pad_pixels,
    bottom_pad_pixels,
    xr_x_dim,
    xr_y_dim,
):
    """Return spatial window of given pixel size when window partially overlaps input data"""

    dx = np.median(np.diff(xr_data[xr_x_dim].values))
    dy = np.median(np.diff(xr_data[xr_y_dim].values))

    if left_pad_pixels > 0:
        assert right_pad_pixels == 0
        x_sel = np.concatenate(
            [
                xr_data[xr_x_dim].values[0] - np.arange(left_pad_pixels, 0, -1) * dx,
                xr_data[xr_x_dim].values[0:right_idx],
            ]
        )
        xr_data = xr_data.isel({xr_x_dim: slice(0, right_idx)}).reindex({xr_x_dim: x_sel})

    elif right_pad_pixels > 0:
        assert left_pad_pixels == 0
        x_sel = np.concatenate(
            [
                xr_data[xr_x_dim].values[left_idx:],
                xr_data[xr_x_dim].values[-1] + np.arange(1, right_pad_pixels + 1) * dx,
            ]
        )
        xr_data = xr_data.isel({xr_x_dim: slice(left_idx, None)}).reindex({xr_x_dim: x_sel})

    else:
        xr_data = xr_data.isel({xr_x_dim: slice(left_idx, right_idx)})

    if top_pad_pixels > 0:
        assert bottom_pad_pixels == 0
        y_sel = np.concatenate(
            [
                xr_data[xr_y_dim].values[0] - np.arange(top_pad_pixels, 0, -1) * dy,
                xr_data[xr_y_dim].values[0:bottom_idx],
            ]
        )
        xr_data = xr_data.isel({xr_y_dim: slice(0, bottom_idx)}).reindex({xr_y_dim: y_sel})

    elif bottom_pad_pixels > 0:
        assert top_pad_pixels == 0
        y_sel = np.concatenate(
            [
                xr_data[xr_y_dim].values[top_idx:],
                xr_data[xr_y_dim].values[-1] + np.arange(1, bottom_pad_pixels + 1) * dy,
            ]
        )
        xr_data = xr_data.isel({xr_y_dim: slice(top_idx, None)}).reindex({xr_x_dim: y_sel})

    else:
        xr_data = xr_data.isel({xr_y_dim: slice(top_idx, bottom_idx)})

    return xr_data


def slice_spatial_pixel_window_from_xarray(
    xr_data, center_idx, width_pixels, height_pixels, xr_x_dim, xr_y_dim, allow_partial_slice
):
    """Select a spatial slice from an xarray object

    Args:
        xr_data: Xarray object
        center_idx: Location object describing the centre of the window
        width_pixels: Window with in pixels
        height_pixels: Window height in pixels
        xr_x_dim: Name of the x-dimension in the xr_data
        xr_y_dim: Name of the y-dimension in the xr_data
        allow_partial_slice: Whether to allow a partially filled window
    """
    half_width = width_pixels // 2
    half_height = height_pixels // 2

    left_idx = int(center_idx.x - half_width)
    right_idx = int(center_idx.x + half_width)
    top_idx = int(center_idx.y - half_height)
    bottom_idx = int(center_idx.y + half_height)

    data_width_pixels = len(xr_data[xr_x_dim])
    data_height_pixels = len(xr_data[xr_y_dim])

    left_pad_required = left_idx < 0
    right_pad_required = right_idx >= data_width_pixels
    top_pad_required = top_idx < 0
    bottom_pad_required = bottom_idx >= data_height_pixels

    pad_required = any(
        [left_pad_required, right_pad_required, top_pad_required, bottom_pad_required]
    )

    if pad_required:
        if allow_partial_slice:
            left_pad_pixels = (-left_idx) if left_pad_required else 0
            right_pad_pixels = (right_idx - (data_width_pixels - 1)) if right_pad_required else 0
            top_pad_pixels = (-top_idx) if top_pad_required else 0
            bottom_pad_pixels = (
                (bottom_idx - (data_height_pixels - 1)) if bottom_pad_required else 0
            )

            xr_data = _slice_patial_spatial_pixel_window_from_xarray(
                xr_data,
                left_idx,
                right_idx,
                top_idx,
                bottom_idx,
                left_pad_pixels,
                right_pad_pixels,
                top_pad_pixels,
                bottom_pad_pixels,
                xr_x_dim,
                xr_y_dim,
            )
        else:
            raise ValueError(
                f"Window for location {center_idx} not available. Missing (left, right, top, "
                f"bottom) pixels  = ({left_pad_required}, {right_pad_required}, "
                f"{top_pad_required}, {bottom_pad_required}). "
                f"You may wish to set `allow_partial_slice=True`"
            )

    else:
        xr_data = xr_data.isel(
            {
                xr_x_dim: slice(left_idx, right_idx),
                xr_y_dim: slice(top_idx, bottom_idx),
            }
        )

    assert len(xr_data[xr_x_dim]) == width_pixels, (
        f"Expected x-dim len {width_pixels} got {len(xr_data[xr_x_dim])} "
        f"for location {center_idx} for slice {left_idx}:{right_idx}"
    )
    assert len(xr_data[xr_y_dim]) == height_pixels, (
        f"Expected y-dim len {height_pixels} got {len(xr_data[xr_y_dim])} "
        f"for location {center_idx} for slice {top_idx}:{bottom_idx}"
    )

    return xr_data


# ---------------------------- main functions for slicing ---------------------------


def select_spatial_slice_pixels(
    xr_data: Union[xr.Dataset, xr.DataArray],
    location: Location,
    roi_width_pixels: int,
    roi_height_pixels: int,
    allow_partial_slice: bool = False,
    location_idx_name: Optional[str] = None,
):
    """
    Select spatial slice based off pixels from location point of interest

    If `allow_partial_slice` is set to True, then slices may be made which intersect the border
    of the input data. The additional x and y cordinates that would be required for this slice
    are extrapolated based on the average spacing of these coordinates in the input data.
    However, currently slices cannot be made where the centre of the window is outside of the
    input data.

    Args:
        xr_data: Xarray DataArray or Dataset to slice from
        location: Location of interest
        roi_height_pixels: ROI height in pixels
        roi_width_pixels: ROI width in pixels
        allow_partial_slice: Whether to allow a partial slice.
        location_idx_name: Name for location index of unstructured grid data,
            None if not relevant
    """

    xr_coords, xr_x_dim, xr_y_dim = spatial_coord_type(xr_data)
    if location_idx_name is not None:
        selected = _get_points_from_unstructured_grids(
            xr_data=xr_data,
            location=location,
            location_idx_name=location_idx_name,
            num_points=roi_width_pixels * roi_height_pixels,
        )
    else:
        if xr_coords == "geostationary":
            center_idx: Location = _get_idx_of_pixel_closest_to_poi_geostationary(
                xr_data=xr_data,
                center_osgb=location,
            )
        else:
            center_idx: Location = _get_idx_of_pixel_closest_to_poi(
                xr_data=xr_data,
                location=location,
            )

        selected = slice_spatial_pixel_window_from_xarray(
            xr_data,
            center_idx,
            roi_width_pixels,
            roi_height_pixels,
            xr_x_dim,
            xr_y_dim,
            allow_partial_slice=allow_partial_slice,
        )

    return selected