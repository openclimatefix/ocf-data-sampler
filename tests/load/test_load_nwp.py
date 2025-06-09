import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert not da.coords["init_time_utc"].isnull().any()
    assert not da.coords["step"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["x_osgb"].isnull().any()
    assert not da.coords["y_osgb"].isnull().any()
    expected_init_time_freq = pd.to_timedelta("3 hours")
    init_time_diffs = da.coords["init_time_utc"].diff("init_time_utc")
    if len(init_time_diffs) > 0:
        assert (init_time_diffs == expected_init_time_freq).all()
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_ecmwf(nwp_ecmwf_zarr_path):
    da = open_nwp(zarr_path=nwp_ecmwf_zarr_path, provider="ecmwf")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert not da.coords["init_time_utc"].isnull().any()
    assert not da.coords["step"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["longitude"].isnull().any()
    assert not da.coords["latitude"].isnull().any()
    expected_init_time_freq = pd.to_timedelta("6 hours")
    init_time_diffs = da.coords["init_time_utc"].diff("init_time_utc")
    if len(init_time_diffs) > 0:
        assert (init_time_diffs == expected_init_time_freq).all()
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_icon_eu(icon_eu_zarr_path):
    da = open_nwp(zarr_path=icon_eu_zarr_path, provider="icon-eu")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (2, 78, 2, 100, 100)
    assert not da.coords["init_time_utc"].isnull().any()
    assert not da.coords["step"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["longitude"].isnull().any()
    assert not da.coords["latitude"].isnull().any()
    expected_init_time_freq = pd.to_timedelta("6 hours")
    init_time_diffs = da.coords["init_time_utc"].diff("init_time_utc")
    if len(init_time_diffs) > 0:
        assert (init_time_diffs == expected_init_time_freq).all()
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_cloudcasting(nwp_cloudcasting_zarr_path):
    da = open_nwp(zarr_path=nwp_cloudcasting_zarr_path, provider="cloudcasting")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")
    assert "area" in da.attrs
    assert da.shape == (2, 12, 3, 100, 100)
    assert not da.coords["init_time_utc"].isnull().any()
    assert not da.coords["step"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["x_geostationary"].isnull().any()
    assert not da.coords["y_geostationary"].isnull().any()
    expected_init_time_freq = pd.to_timedelta("1 hour")
    init_time_diffs = da.coords["init_time_utc"].diff("init_time_utc")
    if len(init_time_diffs) > 0:
        assert (init_time_diffs == expected_init_time_freq).all()
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


@pytest.mark.skip(reason="Fixture 'nwp_gfs_zarr_path' is not yet defined.")
def test_load_gfs_happy_path(nwp_gfs_zarr_path):
    da = open_nwp(zarr_path=nwp_gfs_zarr_path, provider="gfs")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "latitude", "longitude")
    assert not da.coords["init_time_utc"].isnull().any()
    assert not da.coords["step"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["latitude"].isnull().any()
    assert not da.coords["longitude"].isnull().any()
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_load_ukv_bad_dtype():
    """Test validation fails for UKV with bad dtype."""
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1),
        dims=("init_time_utc", "step", "channel", "x_osgb", "y_osgb"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [1.0],
            "channel": ["dswrf"],
            "x_osgb": [0],
            "y_osgb": [0],
        },
    )
    with pytest.raises(TypeError, match="'step' for ukv should be timedelta64"):
        open_nwp(zarr_path="", provider="ukv", _test_data=bad_array)


def test_load_ecmwf_bad_dtype():
    """Test validation fails for ECMWF with bad dtype."""
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1),
        dims=("init_time_utc", "step", "channel", "longitude", "latitude"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "channel": ["t"],
            "longitude": [0],
            "latitude": [50.5],
        },
    )
    with pytest.raises(TypeError, match="'latitude' for ecmwf should be integer"):
        open_nwp(zarr_path="", provider="ecmwf", _test_data=bad_array)


def test_load_icon_eu_bad_dtype():
    """Test validation fails for ICON-EU with bad dtype."""
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1),
        dims=("init_time_utc", "step", "channel", "longitude", "latitude"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "channel": np.array(["t"], dtype=object),
            "latitude": [50.0],
            "longitude": [1],
        },
    )
    with pytest.raises(TypeError, match="'longitude' for icon-eu should be floating"):
        open_nwp(zarr_path="", provider="icon-eu", _test_data=bad_array)


def test_load_cloudcasting_bad_dtype():
    """Test validation fails for Cloudcasting with bad dtype."""
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1),
        dims=("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "channel": [123],
            "x_geostationary": [0],
            "y_geostationary": [0],
        },
    )
    with pytest.raises(TypeError, match="'channel' for cloudcasting should be str_"):
        open_nwp(zarr_path="", provider="cloudcasting", _test_data=bad_array)


def test_load_gfs_bad_dtype():
    """Test validation fails for GFS with bad dtype."""
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1),
        dims=("init_time_utc", "step", "channel", "latitude", "longitude"),
        coords={
            "init_time_utc": [1.23],
            "step": [np.timedelta64(1, "h")],
            "channel": ["t"],
            "latitude": [50.0],
            "longitude": [0.0],
        },
    )
    with pytest.raises(TypeError, match="'init_time_utc' for gfs should be datetime64"):
        open_nwp(zarr_path="", provider="gfs", _test_data=bad_array)
