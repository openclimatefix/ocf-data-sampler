"""Functions for loading locations metadata.

Locations data schema: a CSV file with the following columns:

    location_id: The unique string IDs of the locations
    longitude: The longitudes of the locations
    latitude: The latitudes of the locations

No value may be left blank (i.e. no NaNs).

A CSV is used rather than zarr since this catalogue is small and benefits from being human
readable and hand editable. Additional columns may be included to make the file easier to work with,
but are dropped on load.
"""

import pandas as pd


def open_locations(csv_path: str) -> pd.DataFrame:
    """Open the locations metadata and validate its columns and data types.

    Args:
        csv_path: Path to the locations CSV data

    Returns:
        pd.DataFrame: The locations metadata
    """
    column_dtypes = {
        "location_id": "str",
        "longitude": "float64",
        "latitude": "float64",
    }
    df = pd.read_csv(csv_path, dtype=column_dtypes)

    if missing_columns := set(column_dtypes) - set(df.columns):
        raise ValueError(
            f"Locations data should have columns {list(column_dtypes)}, but the following "
            f"were missing: {missing_columns}",
        )

    # Extra columns are allowed in the file, but are dropped on load to avoid bloat
    df = df[list(column_dtypes)]

    if df.isna().any().any():
        raise ValueError("Locations data must not contain missing values")

    if not df["location_id"].is_unique:
        raise ValueError("location_id must contain unique values")

    return df
