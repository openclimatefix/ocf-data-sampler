"""Functions for loading GSP data."""

from importlib.resources import files

import numpy as np
import pandas as pd
import xarray as xr


def get_gsp_boundaries(version: str) -> pd.DataFrame:
    """Get the GSP boundaries for a given version.

    Args:
        version: Version of the GSP boundaries to use. Options are "20220314" or "20250109".

    Returns:
        pd.DataFrame: The GSP boundaries
    """
    if version not in ["20220314", "20250109"]:
        raise ValueError(
            "Invalid version. Options are '20220314' or '20250109'.",
        )

    return pd.read_csv(
        files("ocf_data_sampler.data").joinpath(f"uk_gsp_locations_{version}.csv"),
        index_col="gsp_id",
    )


def open_gsp(
    zarr_path: str,
    boundaries_version: str = "20220314",
    public: bool = False,
) -> xr.DataArray:
    """Open the GSP data and validates its data types.

    Args:
        zarr_path: Path to the GSP zarr data
        boundaries_version: Version of the GSP boundaries to use. Options are "20220314" or
        "20250109".
        public: Whether the data is public or private.

    Returns:
        xr.DataArray: The opened GSP data
    """
    # Load UK GSP locations
    df_gsp_loc = get_gsp_boundaries(boundaries_version)

    backend_kwargs = {}
    # Open the GSP generation data
    if public:
        backend_kwargs = {"storage_options": {"anon": True}}
        # Currently only compatible with S3 bucket.

    ds = xr.open_dataset(
        zarr_path,
        engine="zarr",
        chunks=None,
        backend_kwargs=backend_kwargs,
    ).rename({"datetime_gmt": "time_utc"})

    if not (ds.gsp_id.isin(df_gsp_loc.index)).all():
        raise ValueError(
            "Some GSP IDs in the GSP generation data are not available in the locations file.",
        )

    # Select the locations by the GSP IDs in the generation data
    df_gsp_loc = df_gsp_loc.loc[ds.gsp_id.values]

    # Add locations and capacities as coordinates for each GSP and datetime
    ds = ds.assign_coords(
        x_osgb=(df_gsp_loc.x_osgb.to_xarray()),
        y_osgb=(df_gsp_loc.y_osgb.to_xarray()),
        nominal_capacity_mwp=ds.installedcapacity_mwp,
        effective_capacity_mwp=ds.capacity_mwp,
    )

    gsp_da = ds.generation_mw

    # Validate data types directly in loading function
    if not np.issubdtype(gsp_da.dtype, np.floating):
        raise TypeError(f"generation_mw should be floating, not {gsp_da.dtype}")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "gsp_id": np.integer,
        "nominal_capacity_mwp": np.floating,
        "effective_capacity_mwp": np.floating,
        "x_osgb": np.floating,
        "y_osgb": np.floating,
    }

    for coord, expected_dtype in coord_dtypes.items():
        if not np.issubdtype(gsp_da.coords[coord].dtype, expected_dtype):
            dtype = gsp_da.coords[coord].dtype
            raise TypeError(f"{coord} should be {expected_dtype.__name__}, not {dtype}")

    return gsp_da
