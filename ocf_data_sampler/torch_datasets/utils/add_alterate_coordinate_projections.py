""""Function for adding more projections to location objects."""

import numpy as np

from ocf_data_sampler.select import Location
from ocf_data_sampler.select.geospatial import convert_coordinates, find_coord_system


def add_alterate_coordinate_projections(
    locations: list[Location],
    datasets_dict: dict,
    primary_coords: str,
) -> list[Location]:
    """Add (in-place) coordinate projections for all dataset to a set of locations.

    Args:
        locations: A list of locations
        datasets_dict: The dataset dict to add projections for
        primary_coords: The primary coords of the locations

    Returns:
        List of locations with all coordinate projections added
    """
    if primary_coords not in ["osgb", "lon_lat"]:
        raise ValueError("Only osbg and lon_lat are currently supported")

    xs, ys = np.array([loc.in_coord_system(primary_coords) for loc in locations]).T

    datasets_list = []
    if "nwp" in datasets_dict:
        datasets_list.extend(datasets_dict["nwp"].values())
    if "sat" in datasets_dict:
        datasets_list.append(datasets_dict["sat"])

    computed_coord_systems = {primary_coords}

    # Find all the coord systems required by all datasets
    for da in datasets_list:

        # Fid the dataset required by this dataset
        coord_system, *_ = find_coord_system(da)

        # Skip if the projections in this coord system have already been computed
        if coord_system not in computed_coord_systems:

            # If using geostationary coords we need to extract the area definition string
            area_string = da.attrs["area"] if coord_system=="geostationary" else None

            new_xs, new_ys = convert_coordinates(
                x=xs,
                y=ys,
                from_coords=primary_coords,
                target_coords=coord_system,
                area_string=area_string,
            )

            # Add the projection to the locations objects
            for x, y, loc in zip(new_xs, new_ys, locations, strict=True):
                loc.add_coord_system(x, y, coord_system)

            computed_coord_systems.add(coord_system)

    # Add lon-lat to start since it is required to compute the solar coords
    if "lon_lat" not in computed_coord_systems:
        new_xs, new_ys = convert_coordinates(
            x=xs,
            y=ys,
            from_coords=primary_coords,
            target_coords="lon_lat",
            area_string=None,
        )

        # Add the projection to the locations objects
        for x, y, loc in zip(new_xs, new_ys, locations, strict=False):
            loc.add_coord_system(x, y, "lon_lat")

    return locations
