"""Refactor legacy site data into a more structured format."""

import pandas as pd
import xarray as xr


def legacy_format(data_ds: xr.Dataset, metadata_df: pd.DataFrame) -> xr.Dataset:
    """Converts old legacy site data into a more structured format.

    This function does three main things:
    1. Renames some columns in the metadata to keep things consistent.
    2. Reshapes site data so that instead of having separate variables for each site,
       we use a `site_id` dimension—makes life easier for analysis.
    3. Adds `capacity_kwp` as a time series so that each site has its capacity info.

    Parameters:
        data_ds (xr.Dataset): The dataset containing legacy site data.
        metadata_df (pd.DataFrame): A DataFrame with metadata about the sites.

    Returns:
        xr.Dataset: Reformatted dataset with `generation_kw` and `capacity_kwp`.
    """
    # Step 1: Rename metadata columns to match the new expected format
    if "system_id" in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"system_id": "site_id"})

    # Convert capacity from megawatts to kilowatts if needed
    if "capacity_megawatts" in metadata_df.columns:
        metadata_df["capacity_kwp"] = metadata_df["capacity_megawatts"] * 1000

    # Quick sanity check to ensure we have what we need
    if "site_id" not in metadata_df.columns or "capacity_kwp" not in metadata_df.columns:
        raise ValueError("Metadata is missing required columns: 'site_id' and 'capacity_kwp'.")

    # Step 2: Transform the dataset
    # Check if we actually have site data in the expected format
    if "0" in data_ds:
        # Convert the dataset into a DataFrame so we can manipulate it more easily
        site_data_df = data_ds.to_dataframe()

        # Create a DataArray for generation data
        generation_da = xr.DataArray(
            data=site_data_df.values,
            coords={
                "time_utc": site_data_df.index.values,
                "site_id": metadata_df["site_id"].values,
            },
            dims=["time_utc", "site_id"],
            name="generation_kw",
        )

        # Step 3: Attach capacity information
        # Map site_ids to their respective capacities
        site_ids = site_data_df.columns
        capacities = metadata_df.set_index("site_id").loc[site_ids, "capacity_kwp"]

        # Broadcast capacities across all timestamps
        capacity_df = pd.DataFrame(
            {site_id: [capacities[site_id]] * len(site_data_df) for site_id in site_ids},
            index=site_data_df.index,
        )

        # Create a DataArray for capacity data
        capacity_da = xr.DataArray(
            data=capacity_df.values,
            coords={
                "time_utc": site_data_df.index.values,
                "site_id": metadata_df["site_id"].values,
            },
            dims=["time_utc", "site_id"],
            name="capacity_kwp",
        )

        # Finally, bundle everything into a single Dataset
        data_ds = xr.Dataset({
            "generation_kw": generation_da,
            "capacity_kwp": capacity_da,
        })

    return data_ds
