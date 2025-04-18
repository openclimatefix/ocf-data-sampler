"""Open GFS Forecast data."""

import logging
import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing

_log = logging.getLogger(__name__)

def open_gfs(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the GFS data.

    Args:
        zarr_path: Path to the Zarr file or list of Zarr files to open.

    Returns:
        Xarray DataArray of the NWP data.

    """
    _log.info(f"Loading NWP GFS data from path(s): {zarr_path}")

    # Validate input type
    if not isinstance(zarr_path, (str, list)):
        raise ValueError("zarr_path must be a string or a list of strings.")

    # Open data
    try:
        gfs: xr.Dataset = open_zarr_paths(zarr_path, time_dim="init_time_utc")
    except Exception as e:
        _log.error(f"Failed to open Zarr path(s) {zarr_path}: {e}")
        raise RuntimeError("Could not open GFS data") from e

    nwp: xr.DataArray = gfs.to_array()

    # Rename dimensions if they exist
    rename_map = {"variable": "channel", "init_time": "init_time_utc"}
    for k, v in rename_map.items():
        if k in nwp.dims:
            nwp = nwp.rename({k: v})

    # Time uniqueness check (left as-is per instructions)
    if "init_time_utc" in nwp.dims or "init_time_utc" in nwp.coords:
        _log.info("Checking that init_time_utc is unique and increasing.")
        check_time_unique_increasing(nwp.init_time_utc)
    else:
        _log.warning("init_time_utc dimension not found, skipping time check.")

    # Spatial coordinate sorting (left as-is per instructions)
    if "longitude" in nwp.dims and "latitude" in nwp.dims:
        _log.info("Ensuring spatial coordinates (longitude, latitude) are increasing.")
        nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")
    else:
        _log.warning("Longitude or latitude dimensions not found, skipping spatial sorting.")

    # Final transpose order
    nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")

    return nwp
