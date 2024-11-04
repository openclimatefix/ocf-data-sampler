""" Helper functions for refactoring legacy site data """


def legacy_format(data_ds, metadata_df):
    """This formats old legacy data to the new format.

    1. This renames the columns in the metadata
    2. Re-formats the site data from data variables named by the site_id to
    a data array with a site_id dimension. Also adds capacity_kwp to the dataset as a time series for each site_id
    """

    if "system_id" in metadata_df.columns:
        metadata_df["site_id"] = metadata_df["system_id"]

    if "capacity_megawatts" in metadata_df.columns:
        metadata_df["capacity_kwp"] = metadata_df["capacity_megawatts"] * 1000

    # only site data has the site_id as data variables.
    # We want to join them all together and create another coordinate called site_id
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