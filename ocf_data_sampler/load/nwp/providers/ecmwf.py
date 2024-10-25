"""ECMWF provider loaders"""
from pathlib import Path
import xarray as xr
from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    make_spatial_coords_increasing,
    get_xr_data_array_from_xr_dataset
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

    # Rename
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        }
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.init_time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="longitude", y_coord="latitude")

    ds = ds.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    
    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
