import pandas as pd
import xarray as xr
import numpy as np

from ocf_data_sampler.config.model import Site


def open_site(sites_config: Site) -> xr.DataArray:

    # Load site generation xr.Dataset
    site_generation_ds = xr.open_dataset(sites_config.file_path)

    # Load site generation data
    metadata_df = pd.read_csv(sites_config.metadata_file_path, index_col="site_id")

    # Ensure metadata aligns with the site_id dimension in data_ds
    metadata_df = metadata_df.reindex(site_generation_ds.site_id.values)

    # Assign coordinates to the Dataset using the aligned metadata
    site_generation_ds = site_generation_ds.assign_coords(
        latitude=("site_id", metadata_df["latitude"].values),
        longitude=("site_id", metadata_df["longitude"].values),
        capacity_kwp=("site_id", metadata_df["capacity_kwp"].values),
    )

    # Sanity checks
    assert np.isfinite(site_generation_ds.capacity_kwp.values).all()
    assert (site_generation_ds.capacity_kwp.values > 0).all()
    assert metadata_df.index.is_unique
    return site_generation_ds.generation_kw
