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

    assert metadata_df.index.is_unique

    # Ensure metadata aligns with the site_id dimension in data_ds
    metadata_df = metadata_df.reindex(generation_ds.site_id.values)

    # Assign coordinates to the Dataset using the aligned metadata
    generation_ds = generation_ds.assign_coords(
        latitude=("site_id", metadata_df["latitude"].values),
        longitude=("site_id", metadata_df["longitude"].values),
        capacity_kwp=("site_id", metadata_df["capacity_kwp"].values),
    )

    # Sanity checks
    assert np.isfinite(generation_ds.capacity_kwp.values).all()
    assert (generation_ds.capacity_kwp.values > 0).all()
    
    return generation_ds.generation_kw
