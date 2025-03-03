"""Open GFS Forecast data"""

import logging
import xarray as xr
from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
_log = logging.getLogger(__name__)


def open_gfs(zarr_path) -> xr.DataArray:
    """
    Opens the GFS data

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    _log.info("Loading NWP GFS data")

    # Open data
    gfs: xr.Dataset = open_zarr_paths(zarr_path, time_dim="init_time_utc")
    nwp: xr.DataArray = gfs.to_array()

    del gfs

    nwp = nwp.rename({"variable": "channel"})
    if "init_time" in nwp.dims:
        nwp = nwp.rename({"init_time": "init_time_utc"})
    nwp = nwp.transpose("init_time_utc", "step", "channel", "latitude", "longitude")

    return nwp