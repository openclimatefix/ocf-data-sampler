"""Open GFS Forecast data."""

import logging

import xarray as xr

from ocf_data_sampler.config.model import NWP
from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing

_log = logging.getLogger(__name__)


def open_gfs(nwp_config: NWP) -> xr.DataArray:
    """Opens the GFS data.

    Args:
        nwp_config: NWP configuration object

    Returns:
        Xarray DataArray of the NWP data
    """
    _log.info("Loading NWP GFS data")

    # Open data
    gfs: xr.Dataset = open_zarr_paths(nwp_config, time_dim="init_time_utc")
    nwp: xr.DataArray = gfs.to_array()

    del gfs

    nwp = nwp.rename({"variable": "channel","init_time": "init_time_utc"})
    check_time_unique_increasing(nwp.init_time_utc)
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")

    nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")

    return nwp
