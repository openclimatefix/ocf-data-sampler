"""UKV provider loaders"""

import xarray as xr

from pathlib import Path


from ocf_data_sampler.load.nwp.providers.utils import (
    open_zarr_paths, check_time_unique_increasing
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

    ds = ds.transpose("init_time", "step", "variable", "y", "x")
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
            "y": "y_osgb",
            "x": "x_osgb",
        }
    )

    # Sanity checks.
    assert ds.y_osgb[0] > ds.y_osgb[1], "UKV must run from top-to-bottom."
    check_time_unique_increasing(ds)
    return ds.UKV


