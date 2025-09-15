import os

import numpy as np
import pytest
from xarray import DataArray

from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_ecmwf(nwp_ecmwf_zarr_path):
    da = open_nwp(zarr_path=nwp_ecmwf_zarr_path, provider="ecmwf")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_icon_eu(icon_eu_zarr_path):
    da = open_nwp(zarr_path=icon_eu_zarr_path, provider="icon-eu")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (2, 78, 3, 100, 100)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_cloudcasting(nwp_cloudcasting_zarr_path):
    da = open_nwp(zarr_path=nwp_cloudcasting_zarr_path, provider="cloudcasting")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")
    assert "area" in da.attrs
    assert da.shape == (2, 12, 3, 100, 100)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


@pytest.mark.skip(reason="Fixture 'nwp_gfs_zarr_path' is not yet defined.")
def test_load_gfs(nwp_gfs_zarr_path):
    da = open_nwp(zarr_path=nwp_gfs_zarr_path, provider="gfs")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "latitude", "longitude")
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_ecmwf_bad_dtype_latitude(tmp_path):
    """Test validation fails for ECMWF with bad latitude dtype."""
    zarr_path = os.path.join(tmp_path, "bad_ecmwf_latitude.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "longitude", "latitude"),
        coords={
            "init_time": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "variable": ["t"],
            "longitude": np.array([0], dtype=np.float32),
            "latitude": [50],
        },
    )
    bad_array.to_zarr(zarr_path)
    with pytest.raises(TypeError, match="'latitude' for ecmwf should be floating"):
        open_nwp(zarr_path=zarr_path, provider="ecmwf")


def test_load_ecmwf_bad_dtype_init_time(tmp_path):
    """Test validation fails for ECMWF with bad init_time_utc dtype."""
    zarr_path = os.path.join(tmp_path, "bad_ecmwf_init_time.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "longitude", "latitude"),
        coords={
            "init_time": [1.23],
            "step": [np.timedelta64(1, "h")],
            "variable": ["t"],
            "longitude": np.array([0], dtype=np.float32),
            "latitude": np.array([50.0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    with pytest.raises((TypeError, AttributeError), match="init_time_utc|datetime64"):
        open_nwp(zarr_path=zarr_path, provider="ecmwf")


def test_load_ecmwf_bad_dtype_step(tmp_path):
    """Test validation fails for ECMWF with bad step dtype."""
    zarr_path = os.path.join(tmp_path, "bad_ecmwf_step.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "longitude", "latitude"),
        coords={
            "init_time": [np.datetime64("2023-01-01")],
            "step": [1.0],
            "variable": ["t"],
            "longitude": np.array([0], dtype=np.float32),
            "latitude": np.array([50.0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    with pytest.raises(TypeError, match="'step' for ecmwf should be timedelta64"):
        open_nwp(zarr_path=zarr_path, provider="ecmwf")
