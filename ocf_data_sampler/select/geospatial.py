"""Geospatial functions"""

from numbers import Number
from typing import Union

import numpy as np
import pyproj
import xarray as xr

# OSGB is also called "OSGB 1936 / British National Grid -- United
# Kingdom Ordnance Survey".  OSGB is used in many UK electricity
# system maps, and is used by the UK Met Office UKV model.  OSGB is a
# Transverse Mercator projection, using 'easting' and 'northing'
# coordinates which are in meters.  See https://epsg.io/27700
OSGB36 = 27700

# WGS84 is short for "World Geodetic System 1984", used in GPS. Uses
# latitude and longitude.
WGS84 = 4326


_osgb_to_lon_lat = pyproj.Transformer.from_crs(
    crs_from=OSGB36, crs_to=WGS84, always_xy=True
).transform
_lon_lat_to_osgb = pyproj.Transformer.from_crs(
    crs_from=WGS84, crs_to=OSGB36, always_xy=True
).transform


def osgb_to_lon_lat(
    x: Union[Number, np.ndarray], y: Union[Number, np.ndarray]
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Change OSGB coordinates to lon, lat.

    Args:
        x: osgb east-west
        y: osgb north-south
    Return: 2-tuple of longitude (east-west), latitude (north-south)
    """
    return _osgb_to_lon_lat(xx=x, yy=y)


def lon_lat_to_osgb(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Change lon-lat coordinates to OSGB.

    Args:
        x: longitude east-west
        y: latitude north-south

    Return: 2-tuple of OSGB x, y
    """
    return _lon_lat_to_osgb(xx=x, yy=y)


def osgb_to_geostationary_area_coords(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
    xr_data: xr.DataArray,
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Loads geostationary area and transformation from OSGB to geostationary coords

    Args:
        x: osgb east-west
        y: osgb north-south
        xr_data: xarray object with geostationary area

    Returns:
        Geostationary coords: x, y
    """
    # Only load these if using geostationary projection
    import pyresample

    area_definition_yaml = xr_data.attrs["area"]

    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=OSGB36, crs_to=geostationary_crs, always_xy=True
    ).transform
    return osgb_to_geostationary(xx=x, yy=y)


def _coord_priority(available_coords):
    if "longitude" in available_coords:
        return "lon_lat", "longitude", "latitude"
    elif "x_geostationary" in available_coords:
        return "geostationary", "x_geostationary", "y_geostationary"
    elif "x_osgb" in available_coords:
        return "osgb", "x_osgb", "y_osgb"
    else:
        raise ValueError(f"Unrecognized coordinate system: {available_coords}")


def spatial_coord_type(ds: xr.DataArray):
    """Searches the data array to determine the kind of spatial coordinates present.

    This search has a preference for the dimension coordinates of the xarray object.

    Args:
        ds: Dataset with spatial coords

    Returns:
        str: The kind of the coordinate system
        x_coord: Name of the x-coordinate
        y_coord: Name of the y-coordinate
    """
    if isinstance(ds, xr.DataArray):
        # Search dimension coords of dataarray
        coords = _coord_priority(ds.xindexes)
    else:
        raise ValueError(f"Unrecognized input type: {type(ds)}")

    return coords
