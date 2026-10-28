"""Convert longitude/latitude to OSGB36 and geostationary coordinates."""

from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import pyproj
from numpy.typing import NDArray

WGS84 = 4326
OSGB36 = 27700


TCoordinateValue = TypeVar("TCoordinateValue", float, NDArray[np.number[Any]])

# Reuse the transformer to avoid rebuilding it each time `lon_lat_to_osgb` is called
_osgb_transformer = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB36, always_xy=True)


def lon_lat_to_osgb(
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


def lon_lat_to_geostationary(
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
    area_definition, = area_spec.values()
    geostationary_crs = pyproj.CRS.from_user_input(area_definition["projection"])
    coord_transformer = pyproj.Transformer.from_crs(
        crs_from=WGS84,
        crs_to=geostationary_crs,
        always_xy=True,
    )
    return coord_transformer.transform(xx=longitude, yy=latitude)
