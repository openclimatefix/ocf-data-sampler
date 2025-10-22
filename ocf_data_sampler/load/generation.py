"""Functions for loading generation data.

Generation data schema: a Zarr file with the following data variables and dimensions/coordinates:

- generation_mw (variable): Floating point representing the generation in MW
- capacity_mwp (variable): Floating point representing the capacity in MWp
- time_utc (dimension/coordinate): Datetime64 representing the UTC timestamps
- location_id (dimension/coordinate): Integer representing the location IDs
- longitude (coordinate): Floating point representing the longitudes of the locations
- latitude (coordinate): Floating point representing the latitudes of the locations

"""

import numpy as np
import xarray as xr


def open_generation(
    zarr_path: str,
    public: bool = False,
) -> xr.DataArray:
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

    ds = ds.assign_coords(
        effective_capacity_mwp=ds.capacity_mwp,
    )

    da = ds.generation_mw

    # Validate data types directly in loading function
    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"generation_mw should be floating, not {da.dtype}")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "location_id": np.integer,
        "effective_capacity_mwp": np.floating,
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
