from pathlib import Path
import xarray as xr

from ocf_data_sampler.load.nwp.providers.ukv import open_ukv
from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs


def open_nwp(zarr_path: Path | str | list[Path] | list[str], provider: str) -> xr.DataArray:
    """Opens NWP Zarr

    Args:
        zarr_path: Path to the Zarr file
        provider: NWP provider
    """

    if provider.lower() == "ukv":
        _open_nwp = open_ukv
    elif provider.lower() == "ecmwf":
        _open_nwp = open_ifs
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return _open_nwp(zarr_path)

