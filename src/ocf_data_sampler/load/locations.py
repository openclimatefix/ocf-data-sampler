"""Functions for loading locations metadata.

Locations data schema: a Zarr file with the following data variables and dimensions/coordinates:

Dimensions: (location_id,)
Data Variables:
    longitude (location_id): The longitudes of the locations
    latitude (location_id): The latitudes of the locations
Coordinates:
    location_id (location_id): The integer IDs of the locations
"""

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.load.conventions import validate_coords


def open_locations(zarr_path: str) -> xr.Dataset:
    """Open and eagerly load the locations metadata and validate its data types.

    Args:
        zarr_path: Path to the locations zarr data

    Returns:
        xr.Dataset: The opened locations metadata
    """
    ds = xr.open_zarr(zarr_path, chunks=None)

    if set(ds.data_vars) != {"longitude", "latitude"}:
        raise ValueError(
            f"Locations data should have variables 'longitude' and 'latitude', "
            f"but found {set(ds.data_vars)} instead."
        )

    validate_coords(ds, {"location_id": np.integer}, source="locations data")

    for var in ("longitude", "latitude"):
        if not np.issubdtype(ds[var].dtype, np.floating):
            raise TypeError(f"{var} in locations data should be floating, not {ds[var].dtype}")

    ds = ds.load()

    assert_values_unique_increasing(ds["location_id"].values, "location_id")

    return ds
