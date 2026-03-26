"""Satellite loader."""
import json

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
        ds = open_zarrs(zarr_path, concat_dim="time", data_source="satellite")
    else:
        ds = open_zarr(zarr_path)

    rename_dict = {
        "variable": "channel",
        "time": "time_utc",
    }

    for old_name, new_name in list(rename_dict.items()):
        if old_name not in ds:
            if new_name in ds:
                del rename_dict[old_name]
            else:
                raise KeyError(f"Expected either '{old_name}' or '{new_name}' to be in dataset")
    ds = ds.rename(rename_dict)

    check_time_unique_increasing(ds.time_utc)
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")

    da = get_xr_data_array_from_xr_dataset(ds)

    # Copy the area attribute if missing
    if "area" not in da.attrs:
        da.attrs["area"] = json.dumps(ds.attrs["area"])

    # Validate data types directly loading function
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"Satellite data should be numeric, not {da.dtype}")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "channel": np.str_,
        "x_geostationary": np.floating,
        "y_geostationary": np.floating,
    }

    for coord, expected_dtype in coord_dtypes.items():
        if not np.issubdtype(da.coords[coord].dtype, expected_dtype):
            dtype = da.coords[coord].dtype
            raise TypeError(f"{coord} should be {expected_dtype.__name__}, not {dtype}")

    return da
