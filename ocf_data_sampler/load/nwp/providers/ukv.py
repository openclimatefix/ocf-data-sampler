"""UKV provider loaders"""

import xarray as xr

from pathlib import Path

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    make_spatial_coords_increasing,
    get_xr_data_array_from_xr_dataset
)


def open_ukv(zarr_path: Path | str | list[Path] | list[str]) -> xr.DataArray:
    """
    Opens the NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    ds = open_zarr_paths(zarr_path)

    # Rename
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
            "x": "x_osgb",
            "y": "y_osgb",
        }
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.init_time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="x_osgb", y_coord="y_osgb")

    ds = ds.transpose("init_time_utc", "step", "channel", "x_osgb", "y_osgb")

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)


