import pandas as pd
import xarray as xr

from ocf_data_sampler.config.model import Site


def open_site(sites_config: Site) -> xr.DataArray:

    # Load site generation xr.Dataset
    data_ds = xr.open_dataset(sites_config.filename)

    # Load site generation data
    metadata_df = pd.read_csv(sites_config.metadata_filename)

    # LEGACY SUPPORT
    data_ds = legacy_format(data_ds, metadata_df)

    metadata_df.set_index("site_id", inplace=True, drop=True)

    # Add coordinates
    ds = data_ds.assign_coords(
        latitude=(metadata_df.latitude.to_xarray()),
        longitude=(metadata_df.longitude.to_xarray()),
        capacity_kwp=data_ds.capacity_kwp,
    )

    return ds.generation_kw


def legacy_format(data_ds, metadata_df):
    """This formats old legacy data to the new format.

    1. This renames the columns in the metadata
    2. Re-formats the site data from data variables named by the site_id to
    a data array with a site_id dimension
    """

    if "system_id" in metadata_df.columns:
        metadata_df["site_id"] = metadata_df["system_id"]

    if "capacity_megawatts" in metadata_df.columns:
        metadata_df["capacity_kwp"] = metadata_df["capacity_megawatts"] * 1000

    # only site data has the site_id as data variables.
    # We want to join them all together and create another variable canned site_id
    if "0" in data_ds:
        gen_df = data_ds.to_dataframe()
        gen_da = xr.DataArray(
            data=gen_df.values,
            coords=(
                ("time_utc", gen_df.index.values),
                ("site_id", metadata_df["site_id"]),
            ),
            name="generation_kw",
        )

        capacity_df = gen_df
        for col in capacity_df.columns:
            capacity_df[col] = metadata_df[metadata_df["site_id"].astype(str) == col][
                "capacity_kwp"
            ].iloc[0]
        capacity_da = xr.DataArray(
            data=capacity_df.values,
            coords=(
                ("time_utc", gen_df.index.values),
                ("site_id", metadata_df["site_id"]),
            ),
            name="capacity_kwp",
        )
        data_ds = xr.Dataset(
            {
                "generation_kw": gen_da,
                "capacity_kwp": capacity_da,
            }
        )
    return data_ds
