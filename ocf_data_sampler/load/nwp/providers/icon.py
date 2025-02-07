"""DWD ICON Loading"""

import pandas as pd
from ocf_data_sampler.load.utils import check_time_unique_increasing, make_spatial_coords_increasing
import xarray as xr
import fsspec

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths

def remove_isobaric_lelvels_from_coords(nwp: xr.Dataset) -> xr.Dataset:
    """
    Removes the isobaric levels from the coordinates of the NWP data
    
    Args:
        nwp: NWP data
    
    Returns:
        NWP data without isobaric levels in the coordinates
    """
    variables_to_drop = [var for var in nwp.data_vars if 'isobaricInhPa' in nwp[var].dims]
    return nwp.drop_vars(["isobaricInhPa"] + variables_to_drop)

def open_icon_eu(zarr_path) -> xr.Dataset:
    """
    Opens the ICON data

    ICON EU Data is on a regular lat/lon grid
    It has data on multiple pressure levels, as well as the surface
    Each of the variables is its own data variable

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path, time_dim="time")
    nwp = nwp.rename({"time": "init_time_utc"})
    # Sanity checks.
    check_time_unique_increasing(nwp.init_time_utc)
    # 0–78 one hour steps, rest 3 hour steps
    nwp = nwp.isel(step=slice(0, 78))
    nwp = remove_isobaric_lelvels_from_coords(nwp)
    nwp = nwp.to_array().rename({"variable": "channel"})
    nwp = nwp.transpose('init_time_utc', 'step', 'channel', 'latitude', 'longitude')
    nwp = make_spatial_coords_increasing(nwp, x_coord="longitude", y_coord="latitude")
    return nwp


def open_icon_global(zarr_path) -> xr.Dataset:
    """
    Opens the ICON data

    ICON Global Data is on an isohedral grid, so the points are not regularly spaced
    It has data on multiple pressure levels, as well as the surface
    Each of the variables is its own data variable

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    # Open the data
    nwp = open_zarr_paths(zarr_path, time_dim="time")
    nwp = nwp.rename({"time": "init_time_utc"})
    # ICON Global archive script didn't define the values to be
    # associated with lat/lon so fixed here
    nwp.coords["latitude"] = (("values",), nwp.latitude.values)
    nwp.coords["longitude"] = (("values",), nwp.longitude.values)
    # Sanity checks.
    time = pd.DatetimeIndex(nwp.init_time_utc)
    assert time.is_unique
    assert time.is_monotonic_increasing
    return nwp
