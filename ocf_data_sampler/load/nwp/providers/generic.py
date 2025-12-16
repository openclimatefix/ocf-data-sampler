"""Generic NWP provider loaders."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_generic(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens generic NWP data with lon/lat coords.

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # LEGACY SUPPORT - different names
    try:
        ds = open_zarr_paths(zarr_path, backend="tensorstore", time_dim="init_time")
    except KeyError:
        ds = open_zarr_paths(zarr_path, backend="tensorstore", time_dim="init_time_utc")

    if "init_time" in ds.coords:
        ds = ds.rename({"init_time": "init_time_utc"})
    if "variable" in ds.coords:
        ds = ds.rename({"variable": "channel"})

    check_time_unique_increasing(ds.init_time_utc)

    ds = make_spatial_coords_increasing(ds, x_coord="longitude", y_coord="latitude")

    ds = ds.transpose("init_time_utc", "step", "channel", "longitude", "latitude")

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
