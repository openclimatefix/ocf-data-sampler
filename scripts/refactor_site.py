import xarray as xr
import pandas as pd


def legacy_format(data_ds, metadata_df):
    """
    Converts legacy site data to the new format.

    1. Renames columns in the metadata DataFrame to align with the new format.
    2. Reformats site data from data variables named by site_id to a DataArray
       with a site_id dimension.
    3. Adds `capacity_kwp` as a time series for each site_id.

    Parameters:
        data_ds (xr.Dataset): Legacy site data with site_id as data variables.
        metadata_df (pd.DataFrame): Metadata for sites.

    Returns:
        xr.Dataset: Reformatted dataset with `generation_kw` and `capacity_kwp`.
    """

    # Rename columns in metadata to match new format
    if "system_id" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"system_id": "site_id"})
    if "capacity_megawatts" in metadata_df.columns:
        metadata_df["capacity_kwp"] = metadata_df["capacity_megawatts"] * 1000

    # Ensure metadata contains required columns
    if "site_id" not in metadata_df.columns or "capacity_kwp" not in metadata_df.columns:
        raise ValueError("Metadata must contain 'site_id' and 'capacity_kwp' columns.")

    # Check if site data contains site_id as data variables
    if "0" in data_ds:
        # Convert data variables to DataFrame
        site_data_df = data_ds.to_dataframe()

        # Create generation DataArray
        generation_da = xr.DataArray(
            data=site_data_df.values,
            coords={
                "time_utc": site_data_df.index.values,
                "site_id": metadata_df["site_id"].values,
            },
            dims=["time_utc", "site_id"],
            name="generation_kw",
        )

        # Create capacity DataArray by broadcasting metadata to match data dimensions
        site_ids = site_data_df.columns
        capacities = metadata_df.set_index("site_id").loc[site_ids, "capacity_kwp"]
        capacity_df = pd.DataFrame(
            {site_id: [capacities[site_id]] * len(site_data_df) for site_id in site_ids},
            index=site_data_df.index,
        )
        capacity_da = xr.DataArray(
            data=capacity_df.values,
            coords={
                "time_utc": site_data_df.index.values,
                "site_id": metadata_df["site_id"].values,
            },
            dims=["time_utc", "site_id"],
            name="capacity_kwp",
        )

        # Combine into a new Dataset
        data_ds = xr.Dataset({"generation_kw": generation_da, "capacity_kwp": capacity_da})

    return data_ds
