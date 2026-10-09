"""Loads all data sources."""

import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.config import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import Coordinate, Location, SourceDict
from ocf_data_sampler.load import open_generation, open_locations, open_nwp, open_satellite
from ocf_data_sampler.load.conventions import validate_step_grid, validate_time_grid
from ocf_data_sampler.spatial import lon_lat_to_geostationary_area_coords, lon_lat_to_osgb

logger = logging.getLogger(__name__)


def _warn_if_not_float32(source: str, data: xr.DataArray) -> None:
    if data.dtype != np.float32:
        logger.warning(
            f"{source} has dtype {data.dtype}; all data sources will be converted to float32 "
            "during materialisation",
        )


def get_dataset_dict(
    config: PVNetDataConfig,
    location_ids: list[int],
) -> SourceDict[xr.DataArray]:
    """Construct dictionary of all of the per-sample input data sources.

    Locations metadata is deliberately excluded - it isn't a per-sample source, so the caller
    loads it separately.

    Args:
        config: PVNetDataConfig configuration object
        location_ids: Requested generation location IDs, in selection order.
    """
    datasets_dict = {}

    # Load generation data if in config
    if config.generation is not None:
        da_gen = open_generation(zarr_path=config.generation.zarr_path)
        _warn_if_not_float32("generation", da_gen)

        validate_time_grid(
            times=da_gen["time_utc"].values,
            resolution=minutes(config.generation.time_resolution_minutes),
            source="generation",
        )

        missing = np.setdiff1d(location_ids, da_gen["location_id"].values)
        if len(missing) > 0:
            raise ValueError(f"Generation data is missing for location IDs: {missing}")

        datasets_dict["generation"] = da_gen.sel(location_id=location_ids)

    # Load NWP data if in config
    if config.nwp:
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in config.nwp.items():
            da_nwp = open_nwp(zarr_path=nwp_config.zarr_path, provider=nwp_config.provider)
            _warn_if_not_float32(f"nwp/{nwp_source}", da_nwp)

            # The NWP init times and steps must be multiples of the configured resolution so that
            # the valid times are aligned to the configured resolution
            validate_time_grid(
                times=da_nwp["init_time_utc"].values,
                resolution=minutes(nwp_config.time_resolution_minutes),
                source=f"nwp/{nwp_source}",
            )

            validate_step_grid(
                steps=da_nwp["step"].values,
                resolution=minutes(nwp_config.time_resolution_minutes),
                source=f"nwp/{nwp_source}",
            )

            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if config.satellite:

        da_sat = open_satellite(config.satellite.zarr_path)
        _warn_if_not_float32("satellite", da_sat)

        validate_time_grid(
            times=da_sat["time_utc"].values,
            resolution=minutes(config.satellite.time_resolution_minutes),
            source="satellite",
        )

        da_sat = da_sat.sel(channel=list(config.satellite.channels))

        datasets_dict["sat"] = da_sat

    return datasets_dict


def get_locations(csv_path: str, exclude_ids: list[int] | None = None) -> list[Location]:
    """Load the locations metadata and build the list of all locations.

    Args:
        csv_path: Path to the locations CSV data
        exclude_ids: Location IDs to drop from the returned locations
    """
    locations_data = open_locations(csv_path)

    if exclude_ids:
        missing_ids = np.setdiff1d(exclude_ids, locations_data["location_id"].values)
        if len(missing_ids) > 0:
            raise ValueError(
                f"Cannot exclude location IDs which are not in the locations data: {missing_ids}",
            )

        locations_data = locations_data[~locations_data["location_id"].isin(exclude_ids)]

        if len(locations_data) == 0:
            raise ValueError("All location IDs in the locations data have been excluded")

    return [
        Location(
            longitude=row.longitude,
            latitude=row.latitude,
            id=int(row.location_id),
        )
        for row in locations_data.itertuples()
    ]


def _find_coord_system(da: xr.DataArray) -> tuple[str, str, str]:
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


def _calculate_source_coordinates(
    longitudes: NDArray[np.float64],
    latitudes: NDArray[np.float64],
    da: xr.DataArray,
) -> list[Coordinate]:
    # Find the coordinate system required by this dataset
    coord_system, x_dim, y_dim = _find_coord_system(da)

    if coord_system == "lon_lat":
        xs, ys = longitudes, latitudes
    elif coord_system == "osgb":
        xs, ys = lon_lat_to_osgb(longitudes, latitudes)
    elif coord_system == "geostationary":
        xs, ys = lon_lat_to_geostationary_area_coords(longitudes, latitudes, da.attrs["area"])

    return [Coordinate(x_dim=x_dim, y_dim=y_dim, x=x, y=y) for x, y in zip(xs, ys, strict=True)]


def add_source_coordinates(
    locations: list[Location],
    datasets_dict: SourceDict[xr.DataArray],
) -> None:
    """Add source coordinates to locations in-place.

    Args:
        locations: A list of locations
        datasets_dict: The dataset dict to calculate source coordinates for
    """
    longitudes = np.array([loc.longitude for loc in locations], dtype=np.float64)
    latitudes = np.array([loc.latitude for loc in locations], dtype=np.float64)

    if (da_sat := datasets_dict.get("sat")) is not None:
        sat_coordinates = _calculate_source_coordinates(longitudes, latitudes, da_sat)
        for loc, sat_coordinate in zip(locations, sat_coordinates, strict=True):
            loc.source_coordinates["sat"] = sat_coordinate

    if "nwp" in datasets_dict:
        for loc in locations:
            loc.source_coordinates["nwp"] = {}

        for nwp_source, da_nwp in datasets_dict["nwp"].items():
            nwp_coordinates = _calculate_source_coordinates(longitudes, latitudes, da_nwp)
            for loc, nwp_coordinate in zip(locations, nwp_coordinates, strict=True):
                loc.source_coordinates["nwp"][nwp_source] = nwp_coordinate
