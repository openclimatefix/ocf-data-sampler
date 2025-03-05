"""Open GFS Forecast data."""

import logging

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing

_log = logging.getLogger(__name__)


def open_gfs(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the GFS data.

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

    nwp = nwp.rename({"variable": "channel","init_time": "init_time_utc"})
    check_time_unique_increasing(nwp.init_time_utc)
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")

    nwp = nwp.transpose("init_time_utc", "step", "channel", "latitude", "longitude")

    return nwp
