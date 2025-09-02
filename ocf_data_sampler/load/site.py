"""Funcitons for loading site data."""

import numpy as np
import pandas as pd
import xarray as xr


def open_site(generation_file_path: str, metadata_file_path: str) -> xr.DataArray:
    """Open a site's generation data and metadata.

    Args:
        generation_file_path: Path to the site generation netcdf data
        metadata_file_path: Path to the site csv metadata

    Returns:
        xr.DataArray: The opened site generation data
    """
    generation_ds = xr.open_dataset(generation_file_path)
    metadata_df = pd.read_csv(metadata_file_path, index_col="site_id")

    if not metadata_df.index.is_unique:
        raise ValueError("site_id is not unique in metadata")

    # Ensure metadata aligns with the site_id dimension in generation_ds
    metadata_df = metadata_df.reindex(generation_ds.site_id.values)

    # Assign coordinates to the Dataset using the aligned metadata
    generation_ds = generation_ds.assign_coords(
        latitude=("site_id", metadata_df["latitude"].values),
        longitude=("site_id", metadata_df["longitude"].values),
        capacity_kwp=("site_id", metadata_df["capacity_kwp"].values),
    )

    # Sanity checks, to prevent inf or negative values
    # Note NaNs are allowed in generation_kw as can have non overlapping time periods for sites
    if np.isinf(generation_ds.generation_kw.values).all():
        raise ValueError("generation_kw contains infinite (+/- inf) values")
    if not (generation_ds.capacity_kwp.values > 0).all():
        raise ValueError("capacity_kwp contains non-positive values")

    site_da = generation_ds.generation_kw

    # Validate data types directly in loading function
    if not np.issubdtype(site_da.dtype, np.floating):
        raise TypeError(f"Generation data should be float, not {site_da.dtype}")


    coord_dtypes = {
    "time_utc": (np.datetime64,),
    "site_id": (np.integer,),
    "capacity_kwp": (np.integer, np.floating),
    "latitude": (np.floating,),
    "longitude": (np.floating,),
}
    for coord, expected_dtypes in coord_dtypes.items():
        if not any(np.issubdtype(site_da.coords[coord].dtype, dt) for dt in expected_dtypes):
            dtype = site_da.coords[coord].dtype
            allowed = ", ".join(dt.__name__ for dt in expected_dtypes)
            raise TypeError(f"{coord} should be one of ({allowed}), not {dtype}")

    # Load the data eagerly into memory by calling compute
    # this makes the dataset faster to sample from, but
    # at the cost of a little extra memory usage
    return site_da.compute()
