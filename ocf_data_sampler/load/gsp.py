from pathlib import Path
import pkg_resources

import pandas as pd
import xarray as xr


def open_gsp(zarr_path: str | Path) -> xr.DataArray:

    # Load GSP generation xr.Dataset
    ds = xr.open_zarr(zarr_path)

    # Rename to standard time name
    ds = ds.rename({"datetime_gmt": "time_utc"})

    # Load UK GSP locations
    df_gsp_loc = pd.read_csv(
        pkg_resources.resource_filename(__name__, "../data/uk_gsp_locations.csv"),
        index_col="gsp_id",
    )

    # Add coordinates
    ds = ds.assign_coords(
        x_osgb=(df_gsp_loc.x_osgb.to_xarray()),
        y_osgb=(df_gsp_loc.y_osgb.to_xarray()),
        nominal_capacity_mwp=ds.installedcapacity_mwp,
        effective_capacity_mwp=ds.capacity_mwp,

    )

    return ds.generation_mw
