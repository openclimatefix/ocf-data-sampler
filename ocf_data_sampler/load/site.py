import pandas as pd
import xarray as xr
import numpy as np

from ocf_data_sampler.config.model import Site


def open_site(sites_config: Site) -> xr.DataArray:

    # Load site generation xr.Dataset
    data_ds = xr.open_dataset(sites_config.file_path)

    # Load site generation data
    metadata_df = pd.read_csv(sites_config.metadata_file_path, index_col="site_id")

    # Add coordinates
    ds = data_ds.assign_coords(
        latitude=(metadata_df.latitude.to_xarray()),
        longitude=(metadata_df.longitude.to_xarray()),
        capacity_kwp=data_ds.capacity_kwp,
    )

    # Sanity checks
    assert np.isfinite(data_ds.capacity_kwp.values).all()
    assert (data_ds.capacity_kwp.values > 0).all()
    assert metadata_df.index.is_unique

    return ds.generation_kw


