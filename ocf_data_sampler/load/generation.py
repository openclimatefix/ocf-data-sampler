"""Functions for loading generation data.

Generation data schema: a Zarr file with the following data variables and dimensions/coordinates:

Dimensions: (time_utc, location_id)
Data Variables:
    generation_mw (time_utc, location_id): float32 representing the generation in MW
    capacity_mwp (time_utc, location_id): float32 representing the capacity in MW peak
Coordinates:
    time_utc (time_utc): datetime64[ns] representing the time in utc
    location_id (location_id): int representing the location IDs
    longitute (location_id): float representing the longitudes of the locations
    latitude (location_id): float representing the latitudes of the locations

"""

import numpy as np
import xarray as xr


def open_generation(zarr_path: str, public: bool = False) -> xr.DataArray:
    """Open and eagerly load the generation data and validates its data types.

    Args:
        zarr_path: Path to the generation zarr data
        public: Whether the data is public or private.

    Returns:
        xr.DataArray: The opened generation data
    """
    backend_kwargs = {}
    # Open the generation data
    if public:
        backend_kwargs = {"storage_options": {"anon": True}}
        # Currently only compatible with S3 bucket.

    ds = xr.open_dataset(
        zarr_path,
        engine="zarr",
        chunks=None,
        backend_kwargs=backend_kwargs,
    )

    ds = ds.assign_coords(capacity_mwp=ds.capacity_mwp)

    da = ds.generation_mw

    # Validate data types
    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"generation_mw should be floating, not {da.dtype}")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "location_id": np.integer,
        "capacity_mwp": np.floating,
        "longitude": np.floating,
        "latitude": np.floating,
    }

    for coord, expected_dtype in coord_dtypes.items():
        if not np.issubdtype(da.coords[coord].dtype, expected_dtype):
            dtype = da.coords[coord].dtype
            raise TypeError(f"{coord} should be {expected_dtype.__name__}, not {dtype}")

    # Below we load the data eagerly into memory - this makes the dataset faster to sample from, but
    # at the cost of a little extra memory usage
    return da.compute()
