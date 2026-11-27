from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.load.locations import open_locations


def _write_csv(tmp_path: Path, df: pd.DataFrame) -> str:
    csv_path = tmp_path / "locations.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_open_locations(locations_csv_path):
    """Test the locations data loader with valid data."""
    df = open_locations(locations_csv_path)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["location_id", "longitude", "latitude"]
    assert all(isinstance(location_id, str) for location_id in df["location_id"])
    assert df["longitude"].dtype == "float64"
    assert df["latitude"].dtype == "float64"
    assert df["location_id"].is_unique


def test_open_locations_unsorted_ids(tmp_path: Path):
    """Unsorted string IDs and their coordinates are preserved in file order."""
    csv_path = _write_csv(
        tmp_path,
        pd.DataFrame(
            {
                "location_id": ["site-7", "site-2", "site-5"],
                "longitude": [1.0, 0.0, 0.5],
                "latitude": [51.0, 50.0, 50.5],
            },
        ),
    )

    df = open_locations(csv_path)
    assert df["location_id"].tolist() == ["site-7", "site-2", "site-5"]
    assert df["longitude"].tolist() == [1.0, 0.0, 0.5]
    assert df["latitude"].tolist() == [51.0, 50.0, 50.5]


def test_open_locations_missing_value(tmp_path: Path):
    """A row with a blank coordinate is rejected rather than loaded as NaN."""
    csv_path = _write_csv(
        tmp_path,
        pd.DataFrame({"location_id": ["loc_1"], "longitude": [0.0], "latitude": [np.nan]}),
    )

    with pytest.raises(ValueError, match="must not contain missing values"):
        open_locations(csv_path)


def test_open_locations_missing_column(tmp_path: Path):
    """Test that open_locations raises a ValueError when a required column is missing."""
    csv_path = _write_csv(
        tmp_path,
        pd.DataFrame({"location_id": ["loc_1", "loc_2"], "longitude": [0.0, 1.0]}),
    )

    with pytest.raises(ValueError, match="Locations data should have columns"):
        open_locations(csv_path)


def test_open_locations_duplicate_ids(tmp_path: Path):
    """Test that open_locations raises a ValueError on repeated location IDs."""
    csv_path = _write_csv(
        tmp_path,
        pd.DataFrame(
            {
                "location_id": ["site-1", "site-1"],
                "longitude": [0.0, 1.0],
                "latitude": [50.0, 51.0],
            },
        ),
    )

    with pytest.raises(ValueError, match="location_id must contain unique values"):
        open_locations(csv_path)
