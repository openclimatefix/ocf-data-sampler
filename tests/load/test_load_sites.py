import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pathlib import Path

from ocf_data_sampler.load.site import open_site


def test_open_site(default_data_site_model):
    """Test the site data loader with valid data."""
    da = open_site(default_data_site_model.file_path, default_data_site_model.metadata_file_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "site_id")
    assert "capacity_kwp" in da.coords
    assert "latitude" in da.coords
    assert "longitude" in da.coords
    assert da.shape == (49, 10)

    assert not da.coords["capacity_kwp"].isnull().any()
    assert not da.coords["latitude"].isnull().any()
    assert not da.coords["longitude"].isnull().any()

    expected_freq = pd.to_timedelta("30 minutes")
    time_diffs = da.coords["time_utc"].diff("time_utc")
    if len(time_diffs) > 0:
        assert (time_diffs == expected_freq).all()

    assert len(np.unique(da.coords["site_id"])) == da.shape[1]


def test_open_site_bad_dtype(tmp_path: Path):
    """Test that open_site raises an error if data types are incorrect."""
    gen_path = tmp_path / "bad_site_gen.nc"
    meta_path = tmp_path / "site_meta.csv"

    bad_ds = xr.Dataset(
        data_vars={
            "generation_kw": (("time_utc", "site_id"), np.random.rand(10, 2))
        },
        coords={
            "time_utc": pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="30T")),
            "site_id": np.array([1.0, 2.0]),
        },
    )
    bad_ds.to_netcdf(gen_path)

    metadata = pd.DataFrame(
        {
            "site_id": [1, 2],
            "latitude": [51.0, 52.0],
            "longitude": [0.0, 1.0],
            "capacity_kwp": [100.0, 120.0],
        }
    )
    metadata.to_csv(meta_path)

    with pytest.raises(TypeError, match="site_id should be integer"):
        open_site(generation_file_path=gen_path, metadata_file_path=meta_path)
