"""Module for opening NWP data."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.cloudcasting import open_cloudcasting
from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.gfs import open_gfs
from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv


def open_nwp(zarr_path: str | list[str],
             provider: str,
             ensemble_member: int | None = None,
             means_path: str | None = None) -> xr.DataArray:
    """Opens NWP zarr.

    Args:
        zarr_path: path to the zarr file
        provider: NWP provider
        ensemble_member: Number of ensemble member to select if NWP ensemble is provided
        means_path: path to zarr with deterministic version to splice with ensemble data if needed

    Returns:
        Xarray DataArray of the NWP data
    """
    provider = provider.lower()

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

    if ensemble_member:
        return _open_nwp(zarr_path, ensemble_member=ensemble_member, means_path=means_path)

    return _open_nwp(zarr_path)
