"""Functions for slicing data around a given location and time."""

import numpy as np

from ocf_data_sampler.common.indexing import get_indices_in_sorted_unique
from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import Coordinate, Location, SourceDict
from ocf_data_sampler.select.dropout import choose_delay
from ocf_data_sampler.select.spatial_slice import (
    select_spatial_slice_pixels,
    select_spatial_slice_pixels_multiple,
)
from ocf_data_sampler.select.time_slice import select_time_slice, select_time_slice_nwp


def slice_sources_by_space(
    source_data: SourceDict[TArray],
    location: Location,
    config: PVNetDataConfig,
) -> SourceDict[TArray]:
    """Slice source arrays around a given location.

    Args:
        source_data: Input source arrays.
        location: The location to sample around
        config: PVNetDataConfig object.

    Returns:
        Sliced source arrays.
    """
    try:
        sliced_source_data: SourceDict[TArray] = {}

        if "nwp" in source_data:
            sliced_source_data["nwp"] = {}

            for nwp_name, nwp_config in config.nwp.items():
                coordinate = location.source_coordinates["nwp"][nwp_name]
                sliced_source_data["nwp"][nwp_name] = select_spatial_slice_pixels(
                    source_data["nwp"][nwp_name],
                    x=coordinate.x,
                    y=coordinate.y,
                    x_dim=coordinate.x_dim,
                    y_dim=coordinate.y_dim,
                    height_pixels=nwp_config.image_size_pixels_height,
                    width_pixels=nwp_config.image_size_pixels_width,
                )

        if "sat" in source_data:
            coordinate = location.source_coordinates["sat"]
            sliced_source_data["sat"] = select_spatial_slice_pixels(
                source_data["sat"],
                x=coordinate.x,
                y=coordinate.y,
                x_dim=coordinate.x_dim,
                y_dim=coordinate.y_dim,
                height_pixels=config.satellite.image_size_pixels_height,
                width_pixels=config.satellite.image_size_pixels_width,
            )

        # Depending on whether this is called before or after time-slicing, the generation data is
        # under a single "generation" key (raw, pre-split) or "generation_input"/"generation_target"
        # (post-split) - slice whichever of these are present by location.
        for key in ("generation", "generation_input", "generation_target"):
            if key not in source_data:
                continue

            location_ids = source_data[key]["location_id"].values
            loc_index = get_indices_in_sorted_unique(location_ids, location.id)

            sliced_source_data[key] = source_data[key].isel(location_id=loc_index)

    except ValueError as error:
        raise ValueError(f"Cannot slice location {location.id!r}: {error}") from error

    return sliced_source_data


def _get_source_spatial_coordinates(
    coordinates: list[Coordinate],
) -> tuple[list[tuple[float, float]], str, str]:
    """Extract paired centres and shared spatial dimensions from source coordinates."""
    if len(coordinates) == 0:
        raise ValueError("`coordinates` is empty - there is no region to cover")

    x_dim = coordinates[0].x_dim
    y_dim = coordinates[0].y_dim
    all_dims = {(coordinate.x_dim, coordinate.y_dim) for coordinate in coordinates}
    if all_dims != {(x_dim, y_dim)}:
        raise ValueError(f"All coordinates must use the same spatial dimensions; found {all_dims}")

    return [(coordinate.x, coordinate.y) for coordinate in coordinates], x_dim, y_dim


def crop_sources_to_locations(
    source_data: SourceDict[TArray],
    locations: list[Location],
    config: PVNetDataConfig,
) -> SourceDict[TArray]:
    """Reduce the spatial extent of the source arrays to only cover the locations.

    Args:
        source_data: Input source arrays.
        locations: List of locations to reduce to
        config: PVNetDataConfig object

    Returns:
        Source arrays with reduced spatial extent.
    """
    try:
        sliced_source_data: SourceDict[TArray] = {}

        if "nwp" in source_data:
            sliced_source_data["nwp"] = {}

            for nwp_name, nwp_config in config.nwp.items():
                source_coordinates = [
                    loc.source_coordinates["nwp"][nwp_name] for loc in locations
                ]
                centres, x_dim, y_dim = _get_source_spatial_coordinates(source_coordinates)
                sliced_source_data["nwp"][nwp_name] = select_spatial_slice_pixels_multiple(
                    source_data["nwp"][nwp_name],
                    centres=centres,
                    x_dim=x_dim,
                    y_dim=y_dim,
                    height_pixels=nwp_config.image_size_pixels_height,
                    width_pixels=nwp_config.image_size_pixels_width,
                )

        if "sat" in source_data:
            source_coordinates = [loc.source_coordinates["sat"] for loc in locations]
            centres, x_dim, y_dim = _get_source_spatial_coordinates(source_coordinates)
            sliced_source_data["sat"] = select_spatial_slice_pixels_multiple(
                source_data["sat"],
                centres=centres,
                x_dim=x_dim,
                y_dim=y_dim,
                height_pixels=config.satellite.image_size_pixels_height,
                width_pixels=config.satellite.image_size_pixels_width,
            )

        if "generation" in source_data:
            sliced_source_data["generation"] = source_data["generation"]

    except ValueError as error:
        raise ValueError(f"Cannot slice all locations: {error}") from error

    return sliced_source_data


def slice_sources_by_time(
    source_data: SourceDict[TArray],
    t0: np.datetime64,
    config: PVNetDataConfig,
) -> SourceDict[TArray]:
    """Slice source arrays around a given t0 time.

    Args:
        source_data: Input source arrays.
        t0: The init-time
        config: PVNetDataConfig object.

    Returns:
        Sliced source arrays.
    """
    sliced_source_data: SourceDict[TArray] = {}

    if "nwp" in source_data:
        sliced_source_data["nwp"] = {}

        for nwp_name, da_nwp in source_data["nwp"].items():
            nwp_config = config.nwp[nwp_name]

            # Add a buffer if we need to diff some of the channels in time
            if any(channel.accumulated for channel in nwp_config.channels):
                interval_end_mins = (
                    nwp_config.interval_end_minutes
                    + nwp_config.time_resolution_minutes
                )
            else:
                interval_end_mins = nwp_config.interval_end_minutes

            sliced_source_data["nwp"][nwp_name] = select_time_slice_nwp(
                da_nwp,
                t0,
                time_resolution=minutes(nwp_config.time_resolution_minutes),
                interval_start=minutes(nwp_config.interval_start_minutes),
                interval_end=minutes(interval_end_mins),
                delay=(
                    None if nwp_config.delay_minute_probabilities is None
                    else choose_delay(nwp_config.delay_minute_probabilities)
                ),
            )

    if "sat" in source_data:

        sliced_source_data["sat"] = select_time_slice(
            source_data["sat"],
            t0,
            time_resolution=minutes(config.satellite.time_resolution_minutes),
            interval_start=minutes(config.satellite.interval_start_minutes),
            interval_end=minutes(config.satellite.interval_end_minutes),
        )

    if "generation" in source_data:
        for key, window_config in (
            ("generation_input", config.generation.input),
            ("generation_target", config.generation.target),
        ):
            if window_config is None:
                continue

            sliced_source_data[key] = select_time_slice(
                source_data["generation"],
                t0,
                time_resolution=minutes(config.generation.time_resolution_minutes),
                interval_start=minutes(window_config.interval_start_minutes),
                interval_end=minutes(window_config.interval_end_minutes),
            )

    return sliced_source_data
