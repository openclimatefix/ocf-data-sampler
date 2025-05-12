"""Open GFS Forecast data."""

import logging

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing

_log = logging.getLogger(__name__)


def open_gfs(zarr_path: str | list[str], public: bool = False) -> xr.DataArray:
    """Opens the GFS data.

    Args:
        zarr_path: Path to the zarr(s) to open
        public: Whether the data is public or private

    Returns:
        Xarray DataArray of the NWP data
    """
    _log.info("Loading NWP GFS data")

    # Open data
    gfs: xr.Dataset = open_zarr_paths(zarr_path, time_dim="init_time_utc", public=public)
    nwp: xr.DataArray = gfs.to_array()
    nwp = nwp.rename({"variable": "channel"})  # `variable` appears when using `to_array`

    del gfs

    check_time_unique_increasing(nwp.init_time_utc)
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")

    nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")

    return nwp
