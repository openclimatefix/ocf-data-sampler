"""Refactor legacy site data into a more structured format."""

import pandas as pd
import xarray as xr

from ocf_data_sampler.select.geospatial import osgb_to_lon_lat


def reformat_legacy_gsp_data(
    gsp_zarr_path: str,
    gsp_metadata_csv_path: str,
    save_path: str,
) -> None:
    """Reformat legacy GSP data to include location coordinates.

    This script reads in legacy GSP generation data (zarr) and location metadata (csv),
    converts OSGB coordinates to latitude and longitude, and saves
    the reformatted dataset with the new location information.
    """
    # Open data sources, make sure they are the same boundary versions
    ds_gsp = xr.open_dataset(gsp_zarr_path, engine="zarr")
    gsp_meta = pd.read_csv(gsp_metadata_csv_path)

    # convert osgb to lat long coordinates
    gsp_meta["longitude"], gsp_meta["latitude"] = osgb_to_lon_lat(
        gsp_meta["x_osgb"].values,
        gsp_meta["y_osgb"].values,
    )

    # Rename variables for consistency and drop unnecessary ones
    ds_gsp = ds_gsp.rename({"datetime_gmt": "time_utc", "gsp_id": "location_id"})
    ds_gsp = ds_gsp.drop_vars("installedcapacity_mwp")

    gsp_meta = gsp_meta.rename(columns={"gsp_id": "location_id"})
    gsp_meta_merge = gsp_meta[["location_id", "longitude", "latitude"]]

    ds_with_meta = ds_gsp.assign_coords(
        latitude=(
            "location_id",
            gsp_meta_merge.set_index("location_id").loc[ds_gsp.location_id.values, "latitude"],
        ),
        longitude=(
            "location_id",
            gsp_meta_merge.set_index("location_id").loc[ds_gsp.location_id.values, "longitude"],
        ),
    )

    # Filter out any times with nans, added this as I saw
    # that there were a small amount (~100) of nans in the new GSP boundaries gen data
    mask = ds_with_meta["generation_mw"].isnull().any(dim="location_id")
    times_to_drop = ds_with_meta.coords["time_utc"].values[mask.values]
    ds_with_meta_dropped = ds_with_meta.drop_sel({"time_utc": times_to_drop})

    ds_with_meta_dropped.drop_encoding().to_zarr(save_path, mode="w")


def reformat_legacy_site_data(
    site_netcdf_path: str,
    site_metadata_csv_path: str,
    save_path: str,
) -> None:
    """Reformat legacy Site data.

    This script reads in legacy Site generation data (netcdf) and location/capacity metadata (csv),
    and saves the reformatted dataset to one zarr with the new location information.
    """
    # Open data sources, make sure they are the same boundary versions
    ds_site = xr.open_dataset(site_netcdf_path)
    site_meta = pd.read_csv(site_metadata_csv_path)

    # convert from kW to MW
    ds_site = ds_site / 1000

    ds_site = ds_site.rename({"site_id": "location_id", "generation_kw": "generation_mw"})

    # Check if capacity is variable or static
    if hasattr(ds_site, "capacity_kwp"):
        ds_site = ds_site.rename({"capacity_kwp": "capacity_mwp"})

    else:
        site_meta["capacity_mwp"] = site_meta["capacity_kwp"] / 1000
        site_meta.drop(columns=["capacity_kwp"], inplace=True)

    site_meta = site_meta.rename(columns={"site_id": "location_id"})

    if hasattr(ds_site, "capacity_mwp"):
        ds_with_meta = ds_site.assign_coords(
            latitude=(
                "location_id",
                site_meta.set_index("location_id").loc[ds_site.location_id.values, "latitude"],
            ),
            longitude=(
                "location_id",
                site_meta.set_index("location_id").loc[ds_site.location_id.values, "longitude"],
            ),
        )
    elif "capacity_mwp" in site_meta.columns:
        capacity = xr.DataArray(
            site_meta.set_index("location_id")["capacity_mwp"],
            dims=["location_id"],
        )

        capacity_broadcasted = capacity.broadcast_like(ds_site)

        ds_with_meta = ds_site.assign(capacity_mwp=capacity_broadcasted)

        ds_with_meta = ds_with_meta.assign_coords(
            latitude=(
                "location_id",
                site_meta.set_index("location_id").loc[ds_with_meta.location_id.values, "latitude"],
            ),
            longitude=(
                "location_id",
                site_meta.set_index("location_id").loc[
                    ds_with_meta.location_id.values,
                    "longitude",
                ],
            ),
        )

    ds_with_meta.drop_encoding().to_zarr(save_path, mode="w")
