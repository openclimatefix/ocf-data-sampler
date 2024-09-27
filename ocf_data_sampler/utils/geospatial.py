"""Geospatial functions"""

from datetime import datetime
from numbers import Number
from typing import Union

import numpy as np
import pandas as pd
import pvlib
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
_geod = pyproj.Geod(ellps="WGS84")


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


def lon_lat_to_geostationary_area_coords(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
    xr_data: Union[xr.Dataset, xr.DataArray],
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Loads geostationary area and change from lon-lat to geostationaery coords

    Args:
        x: Longitude east-west
        y: Latitude north-south
        xr_data: xarray object with geostationary area

    Returns:
        Geostationary coords: x, y
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    try:
        area_definition_yaml = xr_data.attrs["area"]
    except KeyError:
        area_definition_yaml = xr_data.data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    lonlat_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=WGS84,
        crs_to=geostationary_crs,
        always_xy=True,
    ).transform
    return lonlat_to_geostationary(xx=x, yy=y)


def osgb_to_geostationary_area_coords(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
    xr_data: Union[xr.Dataset, xr.DataArray],
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
    import pyproj
    import pyresample

    try:
        area_definition_yaml = xr_data.attrs["area"]
    except KeyError:
        area_definition_yaml = xr_data.data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    osgb_to_geostationary = pyproj.Transformer.from_crs(
        crs_from=OSGB36, crs_to=geostationary_crs, always_xy=True
    ).transform
    return osgb_to_geostationary(xx=x, yy=y)


def geostationary_area_coords_to_osgb(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
    xr_data: Union[xr.Dataset, xr.DataArray],
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Loads geostationary area and change from geostationary coords to OSGB

    Args:
        x: geostationary x coord
        y: geostationary y coord
        xr_data: xarray object with geostationary area

    Returns:
        OSGB x, OSGB y
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    try:
        area_definition_yaml = xr_data.attrs["area"]
    except KeyError:
        area_definition_yaml = xr_data.data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    geostationary_to_osgb = pyproj.Transformer.from_crs(
        crs_from=geostationary_crs, crs_to=OSGB36, always_xy=True
    ).transform
    return geostationary_to_osgb(xx=x, yy=y)


def geostationary_area_coords_to_lonlat(
    x: Union[Number, np.ndarray],
    y: Union[Number, np.ndarray],
    xr_data: Union[xr.Dataset, xr.DataArray],
) -> tuple[Union[Number, np.ndarray], Union[Number, np.ndarray]]:
    """Loads geostationary area and change from geostationary to lon-lat coords

    Args:
        x: geostationary x coord
        y: geostationary y coord
        xr_data: xarray object with geostationary area

    Returns:
        longitude, latitude
    """
    # Only load these if using geostationary projection
    import pyproj
    import pyresample

    try:
        area_definition_yaml = xr_data.attrs["area"]
    except KeyError:
        area_definition_yaml = xr_data.data.attrs["area"]
    geostationary_area_definition = pyresample.area_config.load_area_from_string(
        area_definition_yaml
    )
    geostationary_crs = geostationary_area_definition.crs
    geostationary_to_lonlat = pyproj.Transformer.from_crs(
        crs_from=geostationary_crs, crs_to=WGS84, always_xy=True
    ).transform
    return geostationary_to_lonlat(xx=x, yy=y)


def calculate_azimuth_and_elevation_angle(
    latitude: float, longitude: float, datestamps: list[datetime]
) -> pd.DataFrame:
    """
    Calculation the azimuth angle, and the elevation angle for several datetamps.

    But for one specific lat/lon location

    More details see:
    https://www.celestis.com/resources/faq/what-are-the-azimuth-and-elevation-of-a-satellite/

    Args:
        latitude: latitude of the pv site
        longitude: longitude of the pv site
        datestamps: list of datestamps to calculate the sun angles. i.e the sun moves from east to
            west in the day.

    Returns: Pandas data frame with the index the same as 'datestamps', with columns of
    "elevation" and "azimuth" that have been calculate.

    """
    # get the solor position
    solpos = pvlib.solarposition.get_solarposition(datestamps, latitude, longitude)

    # extract the information we want
    return solpos[["elevation", "azimuth"]]


def move_lon_lat_by_meters(lon, lat, meters_east, meters_north):
    """
    Move a (lon, lat) by a certain number of meters north and east

    Args:
        lon: longitude
        lat: latitude
        meters_east: number of meters to move east
        meters_north: number of meters to move north

    Returns:
        tuple of lon, lat
    """
    new_lon = _geod.fwd(lons=lon, lats=lat, az=90, dist=meters_east)[0]
    new_lat = _geod.fwd(lons=lon, lats=lat, az=0, dist=meters_north)[1]
    return new_lon, new_lat


def _coord_priority(available_coords):
    if "longitude" in available_coords:
        return "lon_lat", "longitude", "latitude"
    elif "x_geostationary" in available_coords:
        return "geostationary", "x_geostationary", "y_geostationary"
    elif "x_osgb" in available_coords:
        return "osgb", "x_osgb", "y_osgb"
    elif "x" in available_coords:
        return "xy", "x", "y"
    else:
        return None, None, None


def spatial_coord_type(ds: xr.Dataset):
    """Searches the dataset to determine the kind of spatial coordinates present.

    This search has a preference for the dimension coordinates of the xarray object. If none of the
    expected coordinates exist in the dimension coordinates, it then searches the non-dimension
    coordinates. See https://docs.xarray.dev/en/latest/user-guide/data-structures.html#coordinates.

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
    elif isinstance(ds, xr.Dataset):
        # Search dimension coords of all variables in dataset
        coords = _coord_priority(set([v for k in ds.keys() for v in list(ds[k].xindexes)]))
    else:
        raise ValueError(f"Unrecognized input type: {type(ds)}")

    if coords == (None, None, None):
        # If no dimension coords found, search non-dimension coords
        coords = _coord_priority(list(ds.coords))

    return coords
