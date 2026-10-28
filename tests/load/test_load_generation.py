from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.generation import open_generation
from tests.conftest import LOCATION_IDS


def test_open_generation(generation_zarr_path, ds_generation):
    """Test the generation data loader with valid data."""
    da = open_generation(generation_zarr_path)

    assert isinstance(da, xr.DataArray)
    # Unlike other sources, generation data should be in memory after loading
    assert da.variable._in_memory
    assert da.dims == ("time_utc", "location_id", "gen_param")
    # 24 hours of 30 minute data (inclusive), every catalogued location, capacity + generation
    assert da.shape == (49, len(LOCATION_IDS), 2)
    assert len(np.unique(da.coords["location_id"])) == da.shape[1]
    assert da.dtype == np.float32
    for name in ("capacity_mwp", "generation_mw"):
        np.testing.assert_array_equal(da.sel(gen_param=name).values, ds_generation[name].values)


def test_open_generation_duplicate_location_ids(tmp_path: Path, ds_generation):
    """Duplicate location IDs are rejected even though sorting is not required."""
    ds_generation = ds_generation.isel(location_id=slice(0, 2)).assign_coords(
        location_id=["site-1", "site-1"],
    )
    zarr_path = tmp_path / "generation.zarr"
    ds_generation.to_zarr(zarr_path)

    with pytest.raises(ValueError, match="location_id must contain unique values"):
        open_generation(str(zarr_path))


def test_open_generation_bad_dtype(tmp_path: Path):
    """Test that open_generation raises a TypeError on incorrect data dtypes."""
    zarr_path = tmp_path / "bad_generation.zarr"

    # Create dataset where generation_mw is integer
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (("time_utc", "location_id"), np.random.randint(0, 100, (10, 2))),
            "capacity_mwp": (("location_id",), [90, 110]),
        },
        coords={
            "time_utc": pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="30min")),
            "location_id": ["site-1", "site-2"],
            "longitude": (("location_id",), [0.0, 1.0]),
            "latitude": (("location_id",), [0.0, 1.0]),
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(TypeError, match="generation and capacity values should be floating"):
        open_generation(zarr_path=str(zarr_path))
