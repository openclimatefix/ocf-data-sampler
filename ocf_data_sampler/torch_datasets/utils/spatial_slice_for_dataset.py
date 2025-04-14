"""Functions for selecting data around a given location."""

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_spatial_slice import select_spatial_slice_pixels


def slice_datasets_by_space(
    datasets_dict: dict,
    location: Location,
    config: Configuration,
) -> dict:
    """Slice the dictionary of input data sources around a given location.

    Args:
        datasets_dict: Dictionary of the input data sources
        location: The location to sample around
        config: Configuration object.
    """
    if not set(datasets_dict.keys()).issubset({"nwp", "sat", "gsp", "site"}):
        raise ValueError(
            "'datasets_dict' should only contain keys 'nwp', 'sat', 'gsp', 'site'",
        )

    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        sliced_datasets_dict["nwp"] = {}

        for nwp_key, nwp_config in config.input_data.nwp.items():
            sliced_datasets_dict["nwp"][nwp_key] = select_spatial_slice_pixels(
                datasets_dict["nwp"][nwp_key],
                location,
                height_pixels=nwp_config.image_size_pixels_height,
                width_pixels=nwp_config.image_size_pixels_width,
            )

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        sliced_datasets_dict["sat"] = select_spatial_slice_pixels(
            datasets_dict["sat"],
            location,
            height_pixels=sat_config.image_size_pixels_height,
            width_pixels=sat_config.image_size_pixels_width,
        )

    if "gsp" in datasets_dict:
        sliced_datasets_dict["gsp"] = datasets_dict["gsp"].sel(gsp_id=location.id)

    if "site" in datasets_dict:
        sliced_datasets_dict["site"] = datasets_dict["site"].sel(site_id=location.id)

    return sliced_datasets_dict
