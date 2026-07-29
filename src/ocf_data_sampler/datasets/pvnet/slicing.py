"""Functions for slicing data around a given location and time."""

import numpy as np

from ocf_data_sampler.common.indexing import get_indices_in_sorted_unique
from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.select.spatial_slice import (
    select_spatial_slice_pixels,
    select_spatial_slice_pixels_multiple,
)
from ocf_data_sampler.select.time_slice import select_time_slice, select_time_slice_nwp
from ocf_data_sampler.spatial import Location


def slice_datasets_by_space(
    datasets_dict: SourceDict,
    location: Location,
    config: PVNetDataConfig,
) -> SourceDict:
    """Slice the dictionary of input data sources around a given location.

    Args:
        datasets_dict: Dictionary of the input data sources
        location: The location to sample around
        config: PVNetDataConfig object.

    Returns:
        A dictionary of the sliced input data sources.
    """
    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        sliced_datasets_dict["nwp"] = {}

        for nwp_key, nwp_config in config.nwp.items():
            sliced_datasets_dict["nwp"][nwp_key] = select_spatial_slice_pixels(
                datasets_dict["nwp"][nwp_key],
                location,
                height_pixels=nwp_config.image_size_pixels_height,
                width_pixels=nwp_config.image_size_pixels_width,
            )

    if "sat" in datasets_dict:
        sliced_datasets_dict["sat"] = select_spatial_slice_pixels(
            datasets_dict["sat"],
            location,
            height_pixels=config.satellite.image_size_pixels_height,
            width_pixels=config.satellite.image_size_pixels_width,
        )

    # Depending on whether this is called before or after time-slicing, the generation data is
    # under a single "generation" key (raw, pre-split) or "generation_input"/"generation_target"
    # (post-split) - slice whichever of these are present by location.
    for key in ("generation", "generation_input", "generation_target"):
        if key not in datasets_dict:
            continue

        location_ids = datasets_dict[key]["location_id"].values
        loc_index = get_indices_in_sorted_unique(location_ids, location.id)

        sliced_datasets_dict[key] = datasets_dict[key].isel(location_id=loc_index)

    return sliced_datasets_dict


def reduce_spatial_extent_of_datasets(
    datasets_dict: SourceDict,
    locations: list[Location],
    config: PVNetDataConfig,
) -> SourceDict:
    """Reduce the spatial extent of the datasets to only cover the locations.

    Args:
        datasets_dict: Dictionary of the input data sources
        locations: List of locations to reduce to
        config: PVNetDataConfig object

    Returns:
        A dictionary of the reduced input data sources.
    """
    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        sliced_datasets_dict["nwp"] = {}

        for nwp_key, nwp_config in config.nwp.items():
            sliced_datasets_dict["nwp"][nwp_key] = select_spatial_slice_pixels_multiple(
                datasets_dict["nwp"][nwp_key],
                locations,
                height_pixels=nwp_config.image_size_pixels_height,
                width_pixels=nwp_config.image_size_pixels_width,
            )


    if "sat" in datasets_dict:
        sat_config = config.satellite

        sliced_datasets_dict["sat"] = select_spatial_slice_pixels_multiple(
            datasets_dict["sat"],
            locations,
            height_pixels=sat_config.image_size_pixels_height,
            width_pixels=sat_config.image_size_pixels_width,
        )

    if "generation" in datasets_dict:
        sliced_datasets_dict["generation"] = datasets_dict["generation"]

    return sliced_datasets_dict


def slice_datasets_by_time(
    datasets_dict: SourceDict,
    t0: np.datetime64,
    config: PVNetDataConfig,
) -> SourceDict:
    """Slice the dictionary of input data sources around a given t0 time.

    Args:
        datasets_dict: Dictionary of the input data sources
        t0: The init-time
        config: PVNetDataConfig object.

    Returns:
        A dictionary of the sliced input data sources.
    """
    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        sliced_datasets_dict["nwp"] = {}

        for nwp_key, da_nwp in datasets_dict["nwp"].items():
            nwp_config = config.nwp[nwp_key]

            # Add a buffer if we need to diff some of the channels in time
            if len(nwp_config.accum_channels)>0:
                interval_end_mins = (
                    nwp_config.interval_end_minutes
                    + nwp_config.time_resolution_minutes
                )
            else:
                interval_end_mins = nwp_config.interval_end_minutes

            sliced_datasets_dict["nwp"][nwp_key] = select_time_slice_nwp(
                da_nwp,
                t0,
                time_resolution=minutes(nwp_config.time_resolution_minutes),
                interval_start=minutes(nwp_config.interval_start_minutes),
                interval_end=minutes(interval_end_mins),
                dropout_timedeltas=minutes(nwp_config.dropout_timedeltas_minutes),
                dropout_frac=nwp_config.dropout_fraction,
            )

    if "sat" in datasets_dict:
        sat_config = config.satellite

        sliced_datasets_dict["sat"] = select_time_slice(
            datasets_dict["sat"],
            t0,
            time_resolution=minutes(sat_config.time_resolution_minutes),
            interval_start=minutes(sat_config.interval_start_minutes),
            interval_end=minutes(sat_config.interval_end_minutes),
        )

    if "generation" in datasets_dict:
        generation_config = config.generation
        for key, window_config in (
            ("generation_input", generation_config.input),
            ("generation_target", generation_config.target),
        ):
            if window_config is None:
                continue

            sliced_datasets_dict[key] = select_time_slice(
                datasets_dict["generation"],
                t0,
                time_resolution=minutes(generation_config.time_resolution_minutes),
                interval_start=minutes(window_config.interval_start_minutes),
                interval_end=minutes(window_config.interval_end_minutes),
            )

    return sliced_datasets_dict
