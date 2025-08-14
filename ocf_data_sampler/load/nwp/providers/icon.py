"""DWD ICON Loading."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing


def open_icon_eu(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the ICON data.

    ICON EU Data is now expected to be on a regular lat/lon grid,
    with a 'channel' dimension directly available (as per the updated fixture).
    The 'isobaricInhPa' dimension is expected to be already handled.

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open and check initially
    ds = open_zarr_paths(zarr_path, time_dim="init_time_utc", backend="dask")

    if "icon_eu_data" in ds.data_vars:
        nwp = ds["icon_eu_data"]
    else:
        raise ValueError("Could not find 'icon_eu_data' DataArray in the ICON-EU Zarr file.")

    check_time_unique_increasing(nwp.init_time_utc)

    # 0-78 one hour steps, rest 3 hour steps
    nwp = nwp.isel(step=slice(0, 78))
    nwp = nwp.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")

    return nwp
