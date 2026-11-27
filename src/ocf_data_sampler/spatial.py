"""Detect spatial coordinate systems and convert longitude/latitude into them."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyproj
import xarray as xr
from numpy.typing import NDArray

WGS84 = 4326
OSGB36 = 27700


# Reuse the transformer to avoid rebuilding it each time `lon_lat_to_osgb` is called
_osgb_transformer = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB36, always_xy=True)


def lon_lat_to_osgb[TCoordinateValue: (float, NDArray[np.number[Any]])](
    longitude: TCoordinateValue,
    latitude: TCoordinateValue,
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert lon-lat coordinates to OSGB.

    Args:
        longitude: Longitude in decimal degrees, positive east
        latitude: Latitude in decimal degrees, positive north

    Returns:
        OSGB easting and northing in metres.
    """
    return _osgb_transformer.transform(xx=longitude, yy=latitude)


def lon_lat_to_geostationary[TCoordinateValue: (float, NDArray[np.number[Any]])](
    longitude: TCoordinateValue,
    latitude: TCoordinateValue,
    area_spec: Mapping[str, Any],
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert from lon-lat to geostationary coords.

    Args:
        longitude: Longitude in decimal degrees, positive east
        latitude: Latitude in decimal degrees, positive north
        area_spec: Mapping containing one named geostationary area with a projection entry.

    Returns:
        x_geostationary, y_geostationary
    """
    (area_definition,) = area_spec.values()
    geostationary_crs = pyproj.CRS.from_user_input(area_definition["projection"])
    coord_transformer = pyproj.Transformer.from_crs(
        crs_from=WGS84,
        crs_to=geostationary_crs,
        always_xy=True,
    )
    return coord_transformer.transform(xx=longitude, yy=latitude)


def detect_coordinate_system(da: xr.Dataset | xr.DataArray) -> tuple[str, str, str]:
    """Searches the DataArray-like object to determine the spatial coordinate system.

    Args:
        da: Dataset with spatial coords

    Returns:
        A tuple containing the coordinate system name, x-coordinate name,
        and y-coordinate name.
    """
    # We only look at the dimensional coords. It is possible that other coordinate systems are
    # included as non-dimensional coords
    dimension_names = set(da.dims)

    coord_systems: dict[str, tuple[str, str]] = {
        "lon_lat": ("longitude", "latitude"),
        "geostationary": ("x_geostationary", "y_geostationary"),
        "osgb": ("x_osgb", "y_osgb"),
    }

    matching_coordinate_systems = []

    for coord_name, coord_set in coord_systems.items():
        if set(coord_set) <= dimension_names:
            matching_coordinate_systems.append(coord_name)

    if len(matching_coordinate_systems) == 0:
        raise ValueError(
            f"Did not find any coordinate pairs in the dimensional coords: {dimension_names}",
        )
    elif len(matching_coordinate_systems) > 1:
        raise ValueError(
            f"Found >1 ({matching_coordinate_systems}) coordinate pairs in the dimensional coords: "
            f"{dimension_names}",
        )
    else:
        coord_system_name = matching_coordinate_systems[0]
        return coord_system_name, *coord_systems[coord_system_name]
