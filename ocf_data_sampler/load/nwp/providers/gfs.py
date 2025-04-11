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

    Note:
        - Renaming of dimensions is performed conditionally.
        - The function checks and logs warnings if expected dimensions are missing.
        - Handles datasets with varying structures gracefully.
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
        raise

    nwp: xr.DataArray = gfs.to_array()
    del gfs  # Free memory after converting to DataArray

    # Create a dictionary of renames based on key existence
    renames = {}
    if "variable" in nwp.dims or "variable" in nwp.coords:
        renames["variable"] = "channel"
    if "init_time" in nwp.dims or "init_time" in nwp.coords:
        renames["init_time"] = "init_time_utc"

    # Only perform renaming if there are dimensions to rename
    if renames:
        _log.info(f"Renaming dimensions: {renames}")
        nwp = nwp.rename(renames)

    # Check if init_time_utc exists after potential renaming
    if "init_time_utc" in nwp.dims or "init_time_utc" in nwp.coords:
        _log.info("Checking that init_time_utc is unique and increasing.")
        check_time_unique_increasing(nwp.init_time_utc)
    else:
        _log.warning("init_time_utc dimension not found, skipping time check.")

    # Ensure spatial coordinates are increasing only if they exist
    if "longitude" in nwp.dims and "latitude" in nwp.dims:
        _log.info("Ensuring spatial coordinates (longitude, latitude) are increasing.")
        nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")
    else:
        _log.warning("Longitude or latitude dimensions not found, skipping spatial sorting.")

    # Verify all required dimensions exist before transposing
    required_dims = ["init_time_utc", "step", "channel", "longitude", "latitude"]
    if all(dim in nwp.dims for dim in required_dims):
        _log.info(f"Transposing dimensions into order: {required_dims}")
        nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    else:
        existing_dims = [dim for dim in required_dims if dim in nwp.dims]
        _log.warning(f"Not all required dimensions exist for transpose. Using existing dimensions: {existing_dims}")
        if existing_dims:
            _log.info(f"Transposing using existing dimensions: {existing_dims}")
            nwp = nwp.transpose(*existing_dims)

    return nwp


if __name__ == "__main__":
    # Example usage of the open_gfs function
    example_path = "path/to/your/gfs/zarr"
    
    try:
        result_data_array = open_gfs(example_path)
        print(result_data_array)
    except Exception as e:
        _log.error(f"An error occurred while processing GFS data: {e}")
