"""UKV provider loaders."""

from pathlib import Path

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_cloudcasting(zarr_path: Path | str | list[Path] | list[str]) -> xr.DataArray:
    """Opens the satellite predictions from cloudcasting.

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the cloudcasting data
    """
    # Open the data
    ds = open_zarr_paths(zarr_path)

    # Rename
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        },
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.init_time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")

    ds = ds.transpose("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")

    return get_xr_data_array_from_xr_dataset(ds)
