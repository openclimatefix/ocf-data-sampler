import os

import numpy as np
import pytest
from xarray import DataArray, Dataset

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
    assert da.shape == (2, 78, 2, 100, 100)
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


@pytest.fixture
def bad_ukv_zarr(tmp_path) -> str:
    """Creates a temp Zarr for UKV. Uses raw 'init_time', 'variable', 'x', and 'y' names."""
    zarr_path = os.path.join(tmp_path, "bad_ukv.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "x", "y"),
        coords={
            "init_time": [np.datetime64("2023-01-01")],
            "step": [1.0],
            "variable": ["dswrf"],
            "x": np.array([0], dtype=np.float32),
            "y": np.array([0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    return zarr_path


def test_load_ukv_bad_dtype(bad_ukv_zarr):
    """Test validation fails for UKV with bad dtype by reading from a temp file."""
    with pytest.raises(TypeError, match="'step' for ukv should be timedelta64"):
        open_nwp(zarr_path=bad_ukv_zarr, provider="ukv")


@pytest.fixture
def bad_ecmwf_zarr(tmp_path) -> str:
    """Creates a temp Zarr for ECMWF. Uses raw 'init_time' and 'variable' names."""
    zarr_path = os.path.join(tmp_path, "bad_ecmwf.zarr")
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
    return zarr_path


def test_load_ecmwf_bad_dtype(bad_ecmwf_zarr):
    """Test validation fails for ECMWF with bad dtype by reading from a temp file."""
    with pytest.raises(TypeError, match="'latitude' for ecmwf should be floating"):
        open_nwp(zarr_path=bad_ecmwf_zarr, provider="ecmwf")


@pytest.fixture
def bad_icon_eu_zarr(tmp_path) -> str:
    """Creates a temp Zarr Dataset for ICON-EU with the expected raw structure."""
    zarr_path = os.path.join(tmp_path, "bad_icon_eu.zarr")
    coords = {
        "time": [np.datetime64("2023-01-01")],
        "step": [np.timedelta64(1, "h")],
        "isobaricInhPa": [1000.0],
        "latitude": np.array([50.0], dtype=np.float32),
        "longitude": np.array([1], dtype=np.int64),
    }
    data_vars = {
        "t": (
            ("time", "step", "latitude", "longitude"),
            np.random.rand(1, 1, 1, 1).astype(np.float32),
        ),
    }
    ds = Dataset(data_vars=data_vars, coords=coords)
    ds.to_zarr(zarr_path, consolidated=True)
    return zarr_path


def test_load_icon_eu_bad_dtype(bad_icon_eu_zarr):
    """Test validation fails for ICON-EU with bad dtype by reading from a temp file."""
    with pytest.raises(TypeError, match="'longitude' for icon-eu should be floating"):
        open_nwp(zarr_path=bad_icon_eu_zarr, provider="icon-eu")


@pytest.fixture
def bad_cloudcasting_zarr(tmp_path) -> str:
    """Creates a temp Zarr for Cloudcasting. Uses raw 'init_time' and 'variable' names."""
    zarr_path = os.path.join(tmp_path, "bad_cloudcasting.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "x_geostationary", "y_geostationary"),
        coords={
            "init_time": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "variable": [123],
            "x_geostationary": np.array([0], dtype=np.float32),
            "y_geostationary": np.array([0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    return zarr_path


def test_load_cloudcasting_bad_dtype(bad_cloudcasting_zarr):
    """Test validation fails for Cloudcasting with bad dtype by reading from a temp file."""
    with pytest.raises(TypeError, match="'channel' for cloudcasting should be str_"):
        open_nwp(zarr_path=bad_cloudcasting_zarr, provider="cloudcasting")


@pytest.fixture
def bad_gfs_zarr(tmp_path) -> str:
    """Creates a temp Zarr for GFS. Uses raw 'init_time' and 'variable' names."""
    zarr_path = os.path.join(tmp_path, "bad_gfs.zarr")
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time", "step", "variable", "latitude", "longitude"),
        coords={
            "init_time": [1.23],
            "step": [np.timedelta64(1, "h")],
            "variable": ["t"],
            "latitude": np.array([50.0], dtype=np.float32),
            "longitude": np.array([0.0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    return zarr_path


def test_load_gfs_bad_dtype(bad_gfs_zarr):
    """Test validation fails for GFS with bad dtype by reading from a temp file."""
    with pytest.raises((TypeError, AttributeError), match="init_time_utc|datetime64"):
        open_nwp(zarr_path=bad_gfs_zarr, provider="gfs")
