"""ECMWF provider loaders"""
from pathlib import Path
import xarray as xr
from ocf_dataset_alpha.load.nwp.providers.utils import (
    open_zarr_paths, check_time_unique_increasing
)


def open_ifs(zarr_path: Path | str | list[Path] | list[str]) -> xr.DataArray:
    """
    Opens the ECMWF IFS NWP data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    ds = open_zarr_paths(zarr_path)

    ds = ds.transpose("init_time", "step", "variable", "latitude", "longitude")
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        }
    )
    # Sanity checks.
    check_time_unique_increasing(ds)
    return ds.ECMWF_UK
