"""DWD ICON Loading."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing


def remove_isobaric_lelvels_from_coords(nwp: xr.Dataset) -> xr.Dataset:
    """Removes the isobaric levels from the coordinates of the NWP data.

    Args:
        nwp: NWP data

    Returns:
        NWP data without isobaric levels in the coordinates
    """
    variables_to_drop = [var for var in nwp.data_vars if "isobaricInhPa" in nwp[var].dims]
    return nwp.drop_vars(["isobaricInhPa", *variables_to_drop])


def open_icon_eu(zarr_path: str | list[str]) -> xr.Dataset:
    """Opens the ICON data.

    ICON EU Data is on a regular lat/lon grid
    It has data on multiple pressure levels, as well as the surface
    Each of the variables is its own data variable

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path, time_dim="time")
    nwp = nwp.rename({"time": "init_time_utc"})
    # Sanity checks.
    check_time_unique_increasing(nwp.init_time_utc)
    # 0-78 one hour steps, rest 3 hour steps
    nwp = nwp.isel(step=slice(0, 78))
    nwp = remove_isobaric_lelvels_from_coords(nwp)
    nwp = nwp.to_array().rename({"variable": "channel"})
    nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")
    return nwp
