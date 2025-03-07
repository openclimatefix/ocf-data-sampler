"""Geospatial coordinate transformation functions.

Provides utilities for working with different coordinate systems
commonly used in geospatial applications, particularly for UK-based data.

Supports conversions between:
- OSGB36 (Ordnance Survey Great Britain, easting/northing in meters)
- WGS84 (World Geodetic System, latitude/longitude in degrees)
- Geostationary satellite coordinate systems
"""

import numpy as np
import pyproj
import pyresample
import xarray as xr

# Coordinate Reference System (CRS) identifiers
# OSGB36: UK Ordnance Survey National Grid (easting/northing in meters)
# Refer to - https://epsg.io/27700
OSGB36 = 27700

# WGS84: World Geodetic System 1984 (latitude/longitude in degrees), used in GPS
WGS84 = 4326

# Pre-init Transformer
_osgb_to_lon_lat = pyproj.Transformer.from_crs(
    crs_from=OSGB36,
    crs_to=WGS84,
    always_xy=True,
).transform
_lon_lat_to_osgb = pyproj.Transformer.from_crs(
    crs_from=WGS84,
    crs_to=OSGB36,
    always_xy=True,
).transform


def osgb_to_lon_lat(
    x: float | np.ndarray,
    y: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Change OSGB coordinates to lon-lat.

    Args:
        x: osgb east-west
        y: osgb north-south
    Return: 2-tuple of longitude (east-west), latitude (north-south)
    """
    return _osgb_to_lon_lat(xx=x, yy=y)


def lon_lat_to_osgb(
    x: float | np.ndarray,
    y: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Change lon-lat coordinates to OSGB.

    Args:
        x: longitude east-west
        y: latitude north-south

    Return: 2-tuple of OSGB x, y
    """
    return _lon_lat_to_osgb(xx=x, yy=y)


def lon_lat_to_geostationary_area_coords(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    xr_data: xr.DataArray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transformation from lat-lon to geostationary coords.

    Args:
        longitude: longitude
        latitude: latitude
        xr_data: xarray object with geostationary area

    Returns:
        Geostationary coords: x, y
    """
    return coordinates_to_geostationary_area_coords(longitude, latitude, xr_data, WGS84)


def osgb_to_geostationary_area_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    xr_data: xr.DataArray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transformation from OSGB to geostationary coords.

    Args:
        x: osgb east-west
        y: osgb north-south
        xr_data: xarray object with geostationary area

    Returns:
        Geostationary coords: x, y
    """
    return coordinates_to_geostationary_area_coords(x, y, xr_data, OSGB36)


def coordinates_to_geostationary_area_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    xr_data: xr.DataArray,
    crs_from: int,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transforms to geostationary coords.

    Args:
        x: osgb east-west, or latitude
        y: osgb north-south, or longitude
        xr_data: xarray object with geostationary area
        crs_from: the cordiates system of x,y

    Returns:
        Geostationary coords: x, y
    """
    if crs_from not in [OSGB36, WGS84]:
        raise ValueError(f"Unrecognized coordinate system: {crs_from}")

    area_definition_yaml = xr_data.attrs["area"]

    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml,
    )
    geostationary_crs = geostationary_area_definition.crs
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=crs_from,
        crs_to=geostationary_crs,
        always_xy=True,
    ).transform
    return osgb_to_geostationary(xx=x, yy=y)


def _coord_priority(available_coords: list[str]) -> tuple[str, str, str]:
    """Determines the coordinate system of spatial coordinates present."""
    if "longitude" in available_coords:
        return "lon_lat", "longitude", "latitude"
    elif "x_geostationary" in available_coords:
        return "geostationary", "x_geostationary", "y_geostationary"
    elif "x_osgb" in available_coords:
        return "osgb", "x_osgb", "y_osgb"
    else:
        raise ValueError(f"Unrecognized coordinate system: {available_coords}")


def spatial_coord_type(ds: xr.DataArray) -> tuple[str, str, str]:
    """Searches the data array to determine the kind of spatial coordinates present.

    This search has a preference for the dimension coordinates of the xarray object.

    Args:
        ds: Dataset with spatial coords

    Returns:
        Three strings with:
            1. The kind of the coordinate system
            2. Name of the x-coordinate
            3. Name of the y-coordinate
    """
    if isinstance(ds, xr.DataArray):
        # Search dimension coords of dataarray
        coords = _coord_priority(ds.xindexes)
    else:
        raise ValueError(f"Unrecognized input type: {type(ds)}")

    return coords
