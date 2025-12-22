""""Function for adding more projections to location objects."""

import numpy as np

from ocf_data_sampler.select import Location
from ocf_data_sampler.select.geospatial import convert_coordinates, find_coord_system


def add_alterate_coordinate_projections(
    locations: list[Location],
    datasets_dict: dict,
) -> list[Location]:
    """Add (in-place) coordinate projections for all dataset to a set of locations.

    Args:
        locations: A list of locations
        datasets_dict: The dataset dict to add projections for

    Returns:
        List of locations with all coordinate projections added
    """
    xs, ys = np.array([loc.in_coord_system("lon_lat") for loc in locations]).T

    datasets_list = []
    if "nwp" in datasets_dict:
        datasets_list.extend(datasets_dict["nwp"].values())
    if "sat" in datasets_dict:
        datasets_list.append(datasets_dict["sat"])

    computed_coord_systems = {"lon_lat"}

    # Find all the coord systems required by all datasets
    for da in datasets_list:

        # Find the coordinate system required by this dataset
        coord_system, *_ = find_coord_system(da)

        # Skip if the projections in this coord system have already been computed
        if coord_system not in computed_coord_systems:

            # If using geostationary coords we need to extract the area definition string
            area_string = da.attrs["area"] if coord_system=="geostationary" else None

            new_xs, new_ys = convert_coordinates(
                x=xs,
                y=ys,
                from_coords="lon_lat",
                target_coords=coord_system,
                area_string=area_string,
            )

            # Add the projection to the locations objects
            for x, y, loc in zip(new_xs, new_ys, locations, strict=True):
                loc.add_coord_system(x, y, coord_system)

            computed_coord_systems.add(coord_system)

    return locations
