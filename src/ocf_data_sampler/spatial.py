"""Geospatial coordinate transformations and location utilities.

Supports coordinate systems:
- OSGB36 (Ordnance Survey Great Britain, easting/northing in meters)
- WGS84 (World Geodetic System, longitude/latitude in degrees)
- Geostationary satellite coordinate systems
"""

from collections.abc import Mapping
from typing import Any, TypeVar

import numpy as np
import pyproj
import yaml
from numpy.typing import NDArray
from pyresample.area_config import load_area_from_string

from ocf_data_sampler.common.types import DataArrayLike

ALLOWED_COORD_SYSTEMS = {"osgb", "lon_lat", "geostationary"}

# Coordinate Reference System (CRS) identifiers

# OSGB36: UK Ordnance Survey National Grid (easting/northing in meters) - https://epsg.io/27700
OSGB36 = 27700
# WGS84: World Geodetic System 1984 (longitude/latitude in degrees) - https://epsg.io/4326
WGS84 = 4326


TCoordinateValue = TypeVar("TCoordinateValue", float, NDArray[np.number[Any]])

# Pre-inititiate coordinate Transformer objects
_osgb_to_lon_lat = pyproj.Transformer.from_crs(crs_from=OSGB36, crs_to=WGS84, always_xy=True)
_lon_lat_to_osgb = pyproj.Transformer.from_crs(crs_from=WGS84, crs_to=OSGB36, always_xy=True)


def osgb_to_lon_lat(
    x: TCoordinateValue,
    y: TCoordinateValue,
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert OSGB coordinates to lon-lat.

    Args:
        x: osgb easting
        y: osgb northing

    Returns:
        longitude, latitude
    """
    return _osgb_to_lon_lat.transform(xx=x, yy=y)


def lon_lat_to_osgb(
    x: TCoordinateValue,
    y: TCoordinateValue,
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert lon-lat coordinates to OSGB.

    Args:
        x: longitude
        y: latitude

    Returns:
        x_osgb, y_osgb
    """
    return _lon_lat_to_osgb.transform(xx=x, yy=y)


def _get_geostationary_coord_transform(
    crs_from: int,
    area_spec: str | Mapping[str, Any],
) -> pyproj.transformer.Transformer:
    """Build a transformer from a geostationary area spec.

    Args:
        x: osgb easting, or longitude
        y: osgb northing, or latitude
        crs_from: the coordinates system of x, y
        area_spec: YAML string or mapping containing the geostationary area definition.

    Returns:
        Coordinate Transformer
    """
    if crs_from not in [OSGB36, WGS84]:
        raise ValueError(f"Unrecognized coordinate system: {crs_from}")

    if isinstance(area_spec, Mapping):
        area_spec = yaml.safe_dump(area_spec, sort_keys=False)

    geostationary_crs = load_area_from_string(area_spec).crs

    return pyproj.Transformer.from_crs(
        crs_from=crs_from,
        crs_to=geostationary_crs,
        always_xy=True,
    )


def lon_lat_to_geostationary_area_coords(
    longitude: TCoordinateValue,
    latitude: TCoordinateValue,
    area_spec: str | Mapping[str, Any],
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert from lon-lat to geostationary coords.

    Args:
        longitude: longitude
        latitude: latitude
        area_spec: YAML string or mapping containing the geostationary area definition.

    Returns:
        x_geostationary, y_geostationary
    """
    coord_transformer = _get_geostationary_coord_transform(WGS84, area_spec)
    return coord_transformer.transform(xx=longitude, yy=latitude)


def osgb_to_geostationary_area_coords(
    x: TCoordinateValue,
    y: TCoordinateValue,
    area_spec: str | Mapping[str, Any],
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert from OSGB to geostationary coords.

    Args:
        x: osgb east-west
        y: osgb south-north
        area_spec: YAML string or mapping containing the geostationary area definition.

    Returns:
        x_geostationary, y_geostationary
    """
    coord_transformer = _get_geostationary_coord_transform(OSGB36, area_spec)
    return coord_transformer.transform(xx=x, yy=y)


def find_coord_system(da: DataArrayLike) -> tuple[str, str, str]:
    """Searches the DataArray-like object to determine the spatial coordinate system.

    Args:
        da: Dataset with spatial coords

    Returns:
        A tuple containing the coordinate system name, x-coordinate name,
        and y-coordinate name.
    """
    # We only look at the dimensional coords. It is possible that other coordinate systems are
    # included as non-dimensional coords
    dimensional_coords = set(da.dims)

    coord_systems: dict[str, tuple[str, str]] = {
        "lon_lat": ("longitude", "latitude"),
        "geostationary": ("x_geostationary", "y_geostationary"),
        "osgb": ("x_osgb", "y_osgb"),
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
    x: TCoordinateValue,
    y: TCoordinateValue,
    from_coords: str,
    target_coords: str,
    area_spec: str | Mapping[str, Any] | None = None,
) -> tuple[TCoordinateValue, TCoordinateValue]:
    """Convert x and y coordinates from one coordinate system to another.

    Args:
        x: The x-coordinate to convert.
        y: The y-coordinate to convert.
        from_coords: The coordinate system to convert from.
        target_coords: The coordinate system to convert to
        area_spec: Optional YAML string or mapping containing the geostationary area
            definition. Only used if from_coords or target_coords is "geostationary".

    Returns:
        The converted (x, y) coordinates.
    """
    if from_coords==target_coords:
        return x, y

    match (from_coords, target_coords):

        case ("osgb", "geostationary"):
            if area_spec is None:
                raise ValueError(
                    "The `area_spec` must be provided when converting to geostationary "
                    "coordinates",
                )
            x, y = osgb_to_geostationary_area_coords(x, y, area_spec)

        case ("lon_lat", "geostationary"):
            if area_spec is None:
                raise ValueError(
                    "The `area_spec` must be provided when converting to geostationary "
                    "coordinates",
                )
            x, y = lon_lat_to_geostationary_area_coords(x, y, area_spec)

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


class Location:
    """A spatial location."""

    def __init__(self, x: float, y: float, coord_system: str, id: int | str | None = None) -> None:
        """A spatial location.

        Args:
            x: The easting / left-right location
            y: The northing / down-up location
            coord_system: The coordinate system
            id: The location ID
        """
        self._check_valid_coord_system(coord_system)
        self._projections: dict[str, tuple[float, float]] = {coord_system: (x, y)}
        self.id = id

    def __repr__(self) -> str:
        """Return a readable representation for diagnostics and error messages."""
        coordinates = dict(sorted(self._projections.items()))
        coord_systems = list(coordinates.keys())
        return (
            f"Location(id={self.id!r}, "
            f"coord_systems={coord_systems}, "
            f"coordinates={coordinates})"
        )

    @staticmethod
    def _check_valid_coord_system(coord_system: str) -> None:
        if coord_system not in ALLOWED_COORD_SYSTEMS:
            raise ValueError(f"Coordinate {coord_system} is not supported")

    def in_coord_system(self, coord_system: str) -> tuple[float, float]:
        """Get the location in a specified coordinate system.

        Args:
            coord_system: The desired output coordinate system
        """
        self._check_valid_coord_system(coord_system)

        if coord_system in self._projections:
            return self._projections[coord_system]
        else:
            raise ValueError(
                f"Requested the coodinate in {coord_system}. This has not yet been added. "
                "The current available coordinate systems are "
                f"{list(self._projections.keys())}",
            )

    def add_coord_system(self, x: float, y: float, coord_system: str) -> None:
        """Add the equivalent location in a different coordinate system.

        Args:
            x: The easting / left-right coordinate
            y: The northing / down-up coordinate
            coord_system: The coordinate system name
        """
        self._check_valid_coord_system(coord_system)
        if coord_system in self._projections:
            if not (x, y)==self._projections[coord_system]:
                raise ValueError(
                    f"Tried to re-add coordinate projection {coord_system}, but the supplied"
                    f"coodrinate values ({x}, {y}) do not match the already stored values "
                    f"{self._projections[coord_system]}",
                )
        else:
            self._projections[coord_system] = (x, y)


