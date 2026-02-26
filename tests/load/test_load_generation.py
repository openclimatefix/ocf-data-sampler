from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.generation import open_generation


def test_open_generation(generation_zarr_path):
    """Test the generation data loader with valid data."""
    da = open_generation(generation_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "location_id")
    assert {"capacity_mwp", "longitude", "latitude"}.issubset(da.coords)
    assert da.shape == (49, 318)
    assert len(np.unique(da.coords["location_id"])) == da.shape[1]


def test_open_generation_bad_dtype(tmp_path: Path):
    """Test that open_generation raises a TypeError on incorrect data dtypes."""
    zarr_path = tmp_path / "bad_generation.zarr"

    # Create dataset where generation_mw is integer
    # Use valid location IDs - check against boundaries file passes
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), np.random.randint(0, 100, (10, 2))),
            "capacity_mwp": (("location_id",), [90.0, 110.0]),
        },
        coords={
            "time_utc": pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="30min")),
            "location_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(TypeError, match="generation_mw should be floating"):
        open_generation(zarr_path=zarr_path)


def test_open_generation_bad_dtype_capacity(tmp_path: Path):
    """Test that open_generation raises a TypeError when capacity_mwp is integer."""
    zarr_path = tmp_path / "bad_capacity.zarr"
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), np.random.rand(5, 2).astype(np.float32)),
            "capacity_mwp": (("location_id",), np.array([90, 110])),
        },
        coords={
            "time_utc": pd.date_range("2023-01-01", periods=5, freq="30min"),
            "location_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)
    with pytest.raises(TypeError, match="capacity_mwp should be floating"):
        open_generation(zarr_path=zarr_path)


def test_open_generation_bad_dtype_time(tmp_path: Path):
    """Test that open_generation raises when time_utc is not datetime64."""
    zarr_path = tmp_path / "bad_time.zarr"
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), np.random.rand(5, 2).astype(np.float32)),
            "capacity_mwp": (("location_id",), [90.0, 110.0]),
        },
        coords={
            "time_utc": [1.0, 2.0, 3.0, 4.0, 5.0],
            "location_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)
    with pytest.raises((TypeError, AttributeError), match=r"time_utc|datetime64"):
        open_generation(zarr_path=zarr_path)


def test_open_generation_nan_in_data(tmp_path: Path):
    """Test that all-NaN generation data is surfaced, not silently filled."""
    zarr_path = tmp_path / "nan_generation.zarr"
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (
                ("time_utc", "location_id"),
                np.full((5, 2),
                np.nan,
                dtype=np.float32),
            ),
            "capacity_mwp": (("location_id",), [90.0, 110.0]),
        },
        coords={
            "time_utc": pd.date_range("2023-01-01", periods=5, freq="30min"),
            "location_id": [1, 2],
            "longitude": ("location_id", np.array([-1.0, -2.0], dtype=np.float32)),
            "latitude": ("location_id", np.array([51.0, 52.0], dtype=np.float32)),
        },
    )
    bad_ds.to_zarr(zarr_path)
    da = open_generation(zarr_path=zarr_path)
    assert da.isnull().all()
