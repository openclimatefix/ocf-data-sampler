"""ECMWF provider loaders."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_ifs(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the ECMWF IFS NWP data.

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the NWP data
    """
    ds = open_zarr_paths(zarr_path, backend="tensorstore")

    # LEGACY SUPPORT - rename variable to channel if it exists
    ds = ds.rename({"init_time": "init_time_utc", "variable": "channel"})

    check_time_unique_increasing(ds.init_time_utc)

    ds = make_spatial_coords_increasing(ds, x_coord="longitude", y_coord="latitude")

    ds = ds.transpose("init_time_utc", "step", "channel", "longitude", "latitude")

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
