from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.satellite import open_sat_data


def test_open_satellite(sat_zarr_path):
    """Test the satellite data loader with valid data."""
    da = open_sat_data(zarr_path=sat_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "x_geostationary", "y_geostationary")
    # 288 is 1 days of data at 5 minutes intervals, 12 * 24
    # There are 11 channels
    # There are 100 x 100 pixels
    assert da.shape == (288, 11, 100, 100)

    assert len(np.unique(da.coords["channel"])) == da.shape[1]


def test_open_satellite_bad_dtype(tmp_path: Path):
    """Test that open_sat_data raises an error if a coordinate has the wrong dtype."""
    zarr_path = tmp_path / "bad_sat.zarr"

    # Create dataset with an integer channel coordinate - should be a string
    bad_ds = xr.Dataset(
        data_vars={
            "data": (
                ("time", "variable", "y_geostationary", "x_geostationary"),
                np.random.rand(10, 2, 4, 4),
            ),
        },
        coords={
            "time": pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="5min")),
            "variable": [1, 2],
            "y_geostationary": np.arange(4),
            "x_geostationary": np.arange(4),
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(TypeError, match="channel should be str_"):
        open_sat_data(zarr_path=zarr_path)

def test_open_satellite_icechunk_main(sat_icechunk_path):
    """Test the satellite data loader with Ice Chunk main branch."""
    # Test Ice Chunk path without commit
    da = open_sat_data(zarr_path=sat_icechunk_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "x_geostationary", "y_geostationary")
    # 25 is 2 hours of data at 5 minutes intervals, 2 * 12 + 1
    # There are 11 channels
    # There are 50 x 50 pixels
    assert da.shape == (25, 11, 50, 50)

    assert len(np.unique(da.coords["channel"])) == da.shape[1]


def test_open_satellite_icechunk_commit(sat_icechunk_path):
    """Test the satellite data loader with Ice Chunk commit SHA."""
    import icechunk

    # Get the commit SHA from the repository using the correct function name
    storage = icechunk.local_filesystem_storage(sat_icechunk_path)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    commit_sha = session.snapshot_id

    # Test Ice Chunk path with commit
    icechunk_commit_path = f"{sat_icechunk_path}@{commit_sha}"
    da = open_sat_data(zarr_path=icechunk_commit_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "x_geostationary", "y_geostationary")
    # 25 is 2 hours of data at 5 minutes intervals, 2 * 12 + 1
    # There are 11 channels
    # There are 50 x 50 pixels
    assert da.shape == (25, 11, 50, 50)

    assert len(np.unique(da.coords["channel"])) == da.shape[1]
