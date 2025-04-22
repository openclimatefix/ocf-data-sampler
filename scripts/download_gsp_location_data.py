"""This script downloads the GSP location data from the Neso API and saves it to a CSV file.

This script was used to create the `uk_gsp_locations_20250109.csv` file in the `data` directory.
"""

import io
import os
import tempfile
import zipfile

import geopandas as gpd
import pandas as pd
import requests

SAVE_PATH = "uk_gsp_locations_20250109.csv"

# --- Configuration ---
GSP_REGIONS_URL = (
    "https://api.neso.energy/dataset/2810092e-d4b2-472f-b955-d8bea01f9ec0/"
    "resource/d95e8c1b-9cd9-41dd-aacb-4b53b8c07c20/download/gsp_regions_20250109.zip"
)
# This is the path to the OSBG version of the boundaries. The lon-lats version can be found at:
#   Proj_4326/GSP_regions_4326_20250109.geojson
GSP_REGIONS_GEOJSON_PATH_IN_ZIP = "Proj_27700/GSP_regions_27700_20250109.geojson"
GSP_NAME_MAP_URL = "https://api.pvlive.uk/pvlive/api/v4/gsp_list"
SAVE_PATH = "uk_gsp_locations_20250109.csv"
# --- End Configuration ---


with tempfile.TemporaryDirectory() as tmpdirname:

    # Download the GSP regions
    response_regions = requests.get(GSP_REGIONS_URL, timeout=30)
    response_regions.raise_for_status()

    # Unzip
    with zipfile.ZipFile(io.BytesIO(response_regions.content)) as z:
        geojson_extract_path = os.path.join(tmpdirname, GSP_REGIONS_GEOJSON_PATH_IN_ZIP)
        z.extract(GSP_REGIONS_GEOJSON_PATH_IN_ZIP, tmpdirname)

    # Load the GSP regions
    df_bound = gpd.read_file(geojson_extract_path)

    # Download the GSP name mapping
    response_map = requests.get(GSP_NAME_MAP_URL, timeout=10)
    response_map.raise_for_status()

    # Load the GSP name mapping
    gsp_name_map = response_map.json()
    df_gsp_name_map = (
        pd.DataFrame(data=gsp_name_map["data"], columns=gsp_name_map["meta"])
        .drop("pes_id", axis=1)
    )


def combine_gsps(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Combine GSPs which have been split into mutliple rows."""
    # If only one row for the GSP name then just return the row
    if len(gdf)==0:
        return gdf.iloc[0]

    # If multiple rows for the GSP then get union of the GSP shapes
    else:
        return gpd.GeoSeries(gdf.unary_union, index=["geometry"], crs=gdf.crs)


# Combine GSPs which have been split into multiple rows
df_bound = (
    df_bound.groupby("GSPs")
    .apply(combine_gsps, include_groups=False)
    .reset_index()
)

# Add the PVLive GSP ID for each GSP
df_bound = (
    df_bound.merge(df_gsp_name_map, left_on="GSPs", right_on="gsp_name")
    .drop("GSPs", axis=1)
)

# Add the national GSP - this is the union of all GSPs
national_boundaries = gpd.GeoDataFrame(
    [["NATIONAL", df_bound.unary_union, 0]],
    columns=["gsp_name", "geometry", "gsp_id"],
    crs=df_bound.crs,
)

df_bound = pd.concat([national_boundaries, df_bound], ignore_index=True)

# Add the coordinates for the centroid of each GSP
df_bound["x_osgb"] = df_bound.geometry.centroid.x
df_bound["y_osgb"] = df_bound.geometry.centroid.y

# Reorder columns, sort by gsp_id (increasing) and save
columns = ["gsp_id", "gsp_name", "geometry", "x_osgb", "y_osgb"]
df_bound[columns].sort_values("gsp_id").to_csv(SAVE_PATH, index=False)
