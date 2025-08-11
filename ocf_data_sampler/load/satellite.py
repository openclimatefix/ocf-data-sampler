"""Satellite loader."""
import numpy as np
import xarray as xr

from ocf_data_sampler.load.open_xarray_tensorstore import open_zarr, open_zarrs
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_sat_data(zarr_path: str | list[str]) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types.

    Args:
      zarr_path: Cloud URL or local path pattern, or list of these. If GCS URL,
                 it must start with 'gs://'
    """
    # Open the data
    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim="time")
    else:
        ds = open_zarr(zarr_path)

    check_time_unique_increasing(ds.time)

    ds = ds.rename(
        {
            "variable": "channel",
            "time": "time_utc",
        },
    )

    check_time_unique_increasing(ds.time_utc)
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")

    data_array = get_xr_data_array_from_xr_dataset(ds)

    # Validate data types directly loading function
    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError(f"Satellite data should be numeric, not {data_array.dtype}")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "channel": np.str_,
        "x_geostationary": np.floating,
        "y_geostationary": np.floating,
    }

    for coord, expected_dtype in coord_dtypes.items():
        if not np.issubdtype(data_array.coords[coord].dtype, expected_dtype):
            dtype = data_array.coords[coord].dtype
            raise TypeError(f"{coord} should be {expected_dtype.__name__}, not {dtype}")

    return data_array
