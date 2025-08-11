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
from pyresample.area_config import load_area_from_string
import xarray as xr

# Coordinate Reference System (CRS) identifiers

# OSGB36: UK Ordnance Survey National Grid (easting/northing in meters) - https://epsg.io/27700
OSGB36 = 27700
# WGS84: World Geodetic System 1984 (latitude/longitude in degrees) - https://epsg.io/4326
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
    area_string: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transformation from lat-lon to geostationary coords.

    Args:
        longitude: longitude
        latitude: latitude
        area_string: String containing yaml geostationary area definition to convert to.

    Returns:
        Geostationary coords: x, y
    """
    return coordinates_to_geostationary_area_coords(longitude, latitude, WGS84, area_string)


def osgb_to_geostationary_area_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    area_string: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transformation from OSGB to geostationary coords.

    Args:
        x: osgb east-west
        y: osgb north-south
        area_string: String containing yaml geostationary area definition to convert to.

    Returns:
        Geostationary coords: x, y
    """
    return coordinates_to_geostationary_area_coords(x, y, OSGB36, area_string)


def coordinates_to_geostationary_area_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    crs_from: int,
    area_string: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Loads geostationary area and transforms to geostationary coords.

    Args:
        x: osgb east-west, or latitude
        y: osgb north-south, or longitude
        crs_from: the cordiates system of x, y
        area_string: String containing yaml geostationary area definition to convert to.

    Returns:
        Geostationary coords: x, y
    """
    if crs_from not in [OSGB36, WGS84]:
        raise ValueError(f"Unrecognized coordinate system: {crs_from}")

    geostationary_crs = load_area_from_string(area_string).crs
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=crs_from,
        crs_to=geostationary_crs,
        always_xy=True,
    ).transform
    return osgb_to_geostationary(xx=x, yy=y)



def spatial_coord_type(ds: xr.DataArray) -> tuple[str, str, str]:
    """Searches the data array to determine the kind of spatial coordinates present.

    Args:
        ds: Dataset with spatial coords

    Returns:
        Three strings with:
            1. The kind of the coordinate system
            2. Name of the x-coordinate
            3. Name of the y-coordinate
    """

    dimensional_coords = set(ds.xindexes)

    # Only one coordinate system should exist in the dimensional coords
    x_coords_found = dimensional_coords.intersection({"x_osgb", "longitude", "x_geostationary"})

    if len(x_coords_found)==0:
        raise ValueError(
            f"Did not find any expected x-coords in the dimensional coords: {dimensional_coords}"
        )
    elif len(x_coords_found)>1:
        raise ValueError(
            f"Found >1 potential x-coords in the dimensional coords: {dimensional_coords}"
        )
    else:
        if "longitude" in dimensional_coords:
            return "lon_lat", "longitude", "latitude"
        elif "x_geostationary" in dimensional_coords:
            return "geostationary", "x_geostationary", "y_geostationary"
        elif "x_osgb" in dimensional_coords:
            return "osgb", "x_osgb", "y_osgb"


def convert_coordinates(
    x: float | np.ndarray,
    y: float | np.ndarray,
    from_coords: str,
    target_coords: str,
    area_string: str | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert x and y coordinates to coordinate system matching xarray data.

    Args:
        
        x: The x-coordinate to convert.
        y: The y-coordinate to convert.
        from_coords: The coordinate system to convert from.
        target_coords: The coordinate system to convert to
        area_string: Optional string containing yaml geostationary area definition. Only used if
            from_coords or target_coords is "geostationary"

    Returns:
        The converted (x, y) coordinates.
    """

    if from_coords!=target_coords:

        match (from_coords, target_coords):
            case ("osgb", "geostationary"):
                assert area_string is not None
                x, y = osgb_to_geostationary_area_coords(x, y, area_string)
            case ("lon_lat", "geostationary"):
                assert area_string is not None
                x, y = lon_lat_to_geostationary_area_coords(x, y, area_string)
            case ("osgb", "lon_lat"):
                x, y = osgb_to_lon_lat(x, y)
            case ("lon_lat", "osgb"):
                x, y = lon_lat_to_osgb(x, y)
            case (_, _):
                raise NotImplementedError(
                    f"Conversion from {from_coords} to "
                    f"{target_coords} is not supported",
                )
    return x, y
