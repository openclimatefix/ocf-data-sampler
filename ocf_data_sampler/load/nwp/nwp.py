"""Module for opening NWP data."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv


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
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _open_nwp(zarr_path)

