"""Satellite loader."""

import xarray as xr

from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)

from . open_tensorstore_zarrs import open_zarr, open_zarrs


def open_sat_data(zarr_path: str | list[str]) -> xr.DataArray:
    """Lazily opens the zarr store.

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

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
