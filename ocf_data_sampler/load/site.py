"""Funcitons for loading site data."""

import numpy as np
import pandas as pd
import xarray as xr

import logging

def open_site(generation_file_path: str, metadata_file_path: str, capacity_mode: str) -> xr.Dataset:
    """Open a site's generation data and metadata.

    Args:
        generation_file_path: Path to the site generation netcdf data
        metadata_file_path: Path to the site csv metadata
        capacity_mode: Set to use static or variable capacity

    Returns:
        xr.Dataset: The opened site generation data containing both generation_kw and capacity_kwp
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
    )

    # Use static capacity from metadata file and assign as a coordinate
    if capacity_mode == "static":
        logging.info("Using static capacity from metadata file")
        generation_ds = generation_ds.assign_coords(
            capacity_kwp=("site_id", metadata_df["capacity_kwp"].values)
        )

    # Use variable capacity from generation file and keep as a data variable
    elif capacity_mode == "variable":
        logging.info("Using variable capacity from generation file")

        # Check that capacity is in expected format
        if "capacity_kwp" not in generation_ds:
            raise ValueError("capacity_kwp must exist in generation file when capacity_mode='variable'")

        if generation_ds.capacity_kwp.dims != ("site_id", "time_utc"):
            raise ValueError(
                f"capacity_kwp must have dimensions (site_id, time_utc) when capacity_mode='variable', "
                f"but got dimensions {generation_ds.capacity_kwp.dims}"
            )

    # Sanity checks
    if not np.isfinite(generation_ds.generation_kw.values).all():
        raise ValueError("generation_kw contains non-finite values")
    if not (generation_ds.capacity_kwp.values > 0).all():
        raise ValueError("capacity_kwp contains non-positive values")

    return generation_ds
