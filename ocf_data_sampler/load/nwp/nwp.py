"""Module for opening NWP data."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv
from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu

def open_nwp(zarr_path: str | list[str], provider: str) -> xr.DataArray:
    """Opens NWP zarr.

    Args:
        zarr_path: path to the zarr file
        provider: NWP provider
    """
    if provider.lower() == "ukv":
        _open_nwp = open_ukv
    elif provider.lower() == "ecmwf":
        _open_nwp = open_ifs
    elif provider.lower() == "icon-eu":
        _open_nwp = open_icon_eu
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _open_nwp(zarr_path)
