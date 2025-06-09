from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.gsp import get_gsp_boundaries, open_gsp


@pytest.mark.parametrize("version, expected_length", [("20220314", 318), ("20250109", 332)])
def test_get_gsp_boundaries(version, expected_length):
    """Test the GSP boundary loader."""
    df = get_gsp_boundaries(version)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == expected_length
    assert "x_osgb" in df.columns
    assert "y_osgb" in df.columns

    assert df.index.is_unique


def test_open_gsp_happy_path(uk_gsp_zarr_path):
    """Test the GSP data loader with valid data."""
    da = open_gsp(uk_gsp_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "gsp_id")

    assert "nominal_capacity_mwp" in da.coords
    assert "effective_capacity_mwp" in da.coords
    assert "x_osgb" in da.coords
    assert "y_osgb" in da.coords
    assert da.shape == (49, 318)

    assert not da.coords["nominal_capacity_mwp"].isnull().any()
    assert not da.coords["effective_capacity_mwp"].isnull().any()
    assert not da.coords["x_osgb"].isnull().any()
    assert not da.coords["y_osgb"].isnull().any()

    expected_freq = pd.to_timedelta("30 minutes")
    time_diffs = da.coords["time_utc"].diff("time_utc")
    if len(time_diffs) > 0:
        assert (time_diffs == expected_freq).all()

    assert len(np.unique(da.coords["gsp_id"])) == da.shape[1]


def test_open_gsp_raises_on_bad_dtype(tmp_path: Path):
    """Test that open_gsp raises a TypeError on incorrect data dtypes."""
    zarr_path = tmp_path / "bad_gsp.zarr"

    # Create dataset where generation_mw is integer
    # Use valid GSP IDs - check against boundaries file passes
    bad_ds = xr.Dataset(
        data_vars={
            "generation_mw": (("datetime_gmt", "gsp_id"), np.random.randint(0, 100, (10, 2))),
            "installedcapacity_mwp": (("gsp_id",), [100.0, 120.0]),
            "capacity_mwp": (("gsp_id",), [90.0, 110.0]),
        },
        coords={
            "datetime_gmt": pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="30T")),
            "gsp_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(TypeError, match="generation_mw should be floating"):
        open_gsp(zarr_path=zarr_path)
