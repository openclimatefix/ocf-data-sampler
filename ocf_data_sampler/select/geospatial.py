"""Geospatial coordinate transformation functions.

Provides utilities for working with different coordinate systems

Supports conversions between:
- OSGB36 (Ordnance Survey Great Britain, easting/northing in meters)
- WGS84 (World Geodetic System, latitude/longitude in degrees)
- Geostationary satellite coordinate systems
"""

import numpy as np
import pyproj
import xarray as xr
from pyresample.area_config import load_area_from_string

# Coordinate Reference System (CRS) identifiers

# OSGB36: UK Ordnance Survey National Grid (easting/northing in meters) - https://epsg.io/27700
OSGB36 = 27700
# WGS84: World Geodetic System 1984 (latitude/longitude in degrees) - https://epsg.io/4326
WGS84 = 4326

# Pre-inititiate coordinate Transformer objects
_osgb_to_lon_lat = pyproj.Transformer.from_crs(crs_from=OSGB36, crs_to=WGS84, always_xy=True)
_lon_lat_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB36, always_xy=True)


def osgb_to_lon_lat(
    x: float | np.ndarray,
    y: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert OSGB coordinates to lon-lat.

    Args:
        x: osgb east-west
        y: osgb south-north

    Return: longitude, latitude
    """
    return _osgb_to_lon_lat.transform(xx=x, yy=y)


def lon_lat_to_osgb(
    x: float | np.ndarray,
    y: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert lon-lat coordinates to OSGB.

    Args:
        x: longitude east-west
        y: latitude south-north

    Return: x_osgb, y_osgb
    """
    return _lon_lat_to_osgb.transform(xx=x, yy=y)


def _get_geostationary_coord_transform(
    crs_from: int,
    area_string: str,
) -> pyproj.transformer.Transformer:
    """Loads geostationary area and transforms to geostationary coords.

    Args:
        x: osgb east-west, or latitude
        y: osgb south-north, or longitude
        crs_from: the cordiates system of x, y
        area_string: String containing yaml geostationary area definition to convert to.

    Returns: Coordinate Transformer
    """
    if crs_from not in [OSGB36, WGS84]:
        raise ValueError(f"Unrecognized coordinate system: {crs_from}")

    geostationary_crs = load_area_from_string(area_string).crs

    return pyproj.Transformer.from_crs(
        crs_from=crs_from,
        crs_to=geostationary_crs,
        always_xy=True,
    )


def lon_lat_to_geostationary_area_coords(
    longitude: float | np.ndarray,
    latitude: float | np.ndarray,
    area_string: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert from lon-lat to geostationary coords.

    Args:
        longitude: longitude
        latitude: latitude
        area_string: String containing yaml geostationary area definition to convert to.

    Returns: x_geostationary, y_geostationary
    """
    coord_transformer = _get_geostationary_coord_transform(WGS84, area_string)
    return coord_transformer.transform(xx=longitude, yy=latitude)


def osgb_to_geostationary_area_coords(
    x: float | np.ndarray,
    y: float | np.ndarray,
    area_string: str,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert from OSGB to geostationary coords.

    Args:
        x: osgb east-west
        y: osgb south-north
        area_string: String containing yaml geostationary area definition to convert to.

    Returns: x_geostationary, y_geostationary
    """
    coord_transformer = _get_geostationary_coord_transform(OSGB36, area_string)
    return coord_transformer.transform(xx=x, yy=y)


def find_coord_system(da: xr.DataArray) -> tuple[str, str, str]:
    """Searches the Xarray object to determine the spatial coordinate system.

    Args:
        da: Dataset with spatial coords

    Returns:
        Three strings with:
            1. The kind of the coordinate system
            2. Name of the x-coordinate
            3. Name of the y-coordinate
    """
    # We only look at the dimensional coords. It is possible that other coordinate systems are
    # included as non-dimensional coords
    dimensional_coords = set(da.xindexes)

    coord_systems = {
        "lon_lat": ["longitude", "latitude"],
        "geostationary": ["x_geostationary", "y_geostationary"],
        "osgb": ["x_osgb", "y_osgb"],
    }

    coords_systems_found = []

    for coord_name, coord_set in coord_systems.items():
        if set(coord_set) <= dimensional_coords:
            coords_systems_found.append(coord_name)

    if len(coords_systems_found)==0:
        raise ValueError(
            f"Did not find any coordinate pairs in the dimensional coords: {dimensional_coords}",
        )
    elif len(coords_systems_found)>1:
        raise ValueError(
            f"Found >1 ({coords_systems_found}) coordinate pairs in the dimensional coords: "
            f"{dimensional_coords}",
        )
    else:
        coord_system_name = coords_systems_found[0]
        return coord_system_name, *coord_systems[coord_system_name]


def convert_coordinates(
    x: float | np.ndarray,
    y: float | np.ndarray,
    from_coords: str,
    target_coords: str,
    area_string: str | None = None,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert x and y coordinates from one coordinate system to another.

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
    if from_coords==target_coords:
        return x, y

    if "geostationary" in (from_coords, target_coords) and area_string is not None:
        ValueError("If using geostationary coords the `area_string` must be provided")

    match (from_coords, target_coords):

        case ("osgb", "geostationary"):
            x, y = osgb_to_geostationary_area_coords(x, y, area_string)

        case ("lon_lat", "geostationary"):
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
