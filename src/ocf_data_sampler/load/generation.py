"""Functions for loading generation data.

Generation data schema: a Zarr file with the following data variables and dimensions/coordinates:

Dimensions: (time_utc, location_id)
Data Variables:
    generation_mw (time_utc, location_id): The generation in MW
    capacity_mwp (time_utc, location_id): The capacity in MW peak
Coordinates:
    time_utc (time_utc): The datetimes associated with each generation and capacity value
    location_id (location_id): The integer IDs of the locations
    longitude (location_id): The longitudes of the locations
    latitude (location_id): The latitudes of the locations

"""

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.load.conventions import validate_coords


def open_generation(zarr_path: str) -> xr.DataArray:
    """Open and eagerly load the generation data and validates its data types.

    Args:
        zarr_path: Path to the generation zarr data

    Returns:
        xr.DataArray: The opened generation data
    """
    # Open generation without xarray-tensorstore since it has multiple data-variables
    # TODO: establish if xarray tensorstore could be used here
    ds = xr.open_dataset(zarr_path, engine="zarr", chunks=None)

    if set(ds.data_vars) != {"generation_mw", "capacity_mwp"}:
        raise ValueError(
            f"Generation data should have variables 'generation_mw' and 'capacity_mwp', "
            f"but found {set(ds.data_vars)} instead."
        )

    coord_dtypes = {
        "time_utc": np.datetime64,
        "location_id": np.integer,
        "longitude": np.number,
        "latitude": np.number,
    }
    validate_coords(
        ds,
        coord_dtypes,
        source="generation data",
    )

    # Load the data eagerly into memory - this makes the dataset faster to sample from, but
    # at the cost of a little extra memory usage
    ds = ds.load()

    da = ds.to_dataarray("gen_param").transpose("time_utc", "location_id", "gen_param")

    assert_values_unique_increasing(ds["time_utc"].values, "time_utc")
    assert_values_unique_increasing(ds["location_id"].values, "location_id")

    # Validate data types
    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"generation and capacity values should be floating, not {da.dtype}")

    return da
