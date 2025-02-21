"""Functions for loading GSP data."""

import pandas as pd
import pkg_resources
import xarray as xr


def open_gsp(zarr_path: str) -> xr.DataArray:
    """Open the GSP data.

    Args:
        zarr_path: Path to the GSP zarr data

    Returns:
        xr.DataArray: The opened GSP data
    """
    ds = xr.open_zarr(zarr_path)

    ds = ds.rename({"datetime_gmt": "time_utc"})

    # Load UK GSP locations
    df_gsp_loc = pd.read_csv(
        pkg_resources.resource_filename(__name__, "../data/uk_gsp_locations.csv"),
        index_col="gsp_id",
    )

    # Add locations and capacities as coordinates for each GSP and datetime
    ds = ds.assign_coords(
        x_osgb=(df_gsp_loc.x_osgb.to_xarray()),
        y_osgb=(df_gsp_loc.y_osgb.to_xarray()),
        nominal_capacity_mwp=ds.installedcapacity_mwp,
        effective_capacity_mwp=ds.capacity_mwp,
    )

    return ds.generation_mw
