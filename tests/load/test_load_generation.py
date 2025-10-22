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
    assert {"effective_capacity_mwp", "longitude", "latitude"}.issubset(da.coords)
    assert da.shape == (49, 318)
    assert len(np.unique(da.coords["location_id"])) == da.shape[1]


def test_open_generation_bad_dtype(tmp_path: Path):
    """Test that open_gsp raises a TypeError on incorrect data dtypes."""
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
