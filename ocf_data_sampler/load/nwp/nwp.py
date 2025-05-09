"""Module for opening NWP data."""

import xarray as xr

from ocf_data_sampler.config.model import NWP
from ocf_data_sampler.load.nwp.providers.cloudcasting import open_cloudcasting
from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.gfs import open_gfs
from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv


def open_nwp(nwp_config: NWP) -> xr.DataArray:
    """Opens NWP zarr.

    Args:
        nwp_config: NWP configuration object

    Returns:
        Xarray DataArray of the NWP data
    """
    provider = nwp_config.provider.lower()

    if provider == "ukv":
        _open_nwp = open_ukv
    elif provider == "ecmwf":
        _open_nwp = open_ifs
    elif provider == "icon-eu":
        _open_nwp = open_icon_eu
    elif provider == "gfs":
        _open_nwp = open_gfs
    elif provider == "cloudcasting":
        _open_nwp = open_cloudcasting
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _open_nwp(nwp_config)
