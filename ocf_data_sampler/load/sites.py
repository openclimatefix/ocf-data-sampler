import pandas as pd
import xarray as xr

from ocf_data_sampler.config.model import Sites


def open_sites(sites_config: Sites) -> xr.DataArray:

    # Load site generation xr.Dataset
    data_ds = xr.open_dataset(sites_config.filename)

    # Load site generation data
    metadata_df = pd.read_csv(sites_config.metadata_filename)
    metadata_df.set_index("system_id", inplace=True, drop=True)

    # Add coordinates
    ds = data_ds.assign_coords(
        latitude=(metadata_df.latitude.to_xarray()),
        longitude=(metadata_df.longitude.to_xarray()),
        capacity_kwp=data_ds.capacity_kwp,
    )

    return ds.generation_kw
