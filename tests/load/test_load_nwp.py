import numpy as np
import pandas as pd
from xarray import DataArray

from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert np.issubdtype(da.dtype, np.number)
    assert np.issubdtype(da.coords["init_time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["step"].dtype, np.timedelta64)
    assert np.issubdtype(da.coords["channel"].dtype, np.str_)
    assert np.issubdtype(da.coords["x_osgb"].dtype, np.floating)
    assert np.issubdtype(da.coords["y_osgb"].dtype, np.floating)

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
    assert np.issubdtype(da.dtype, np.number)
    assert np.issubdtype(da.coords["init_time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["step"].dtype, np.timedelta64)
    assert np.issubdtype(da.coords["channel"].dtype, np.str_)
    assert np.issubdtype(da.coords["longitude"].dtype, np.integer)
    assert np.issubdtype(da.coords["latitude"].dtype, np.integer)

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
    assert np.issubdtype(da.dtype, np.number)
    assert np.issubdtype(da.coords["init_time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["step"].dtype, np.timedelta64)
    assert np.issubdtype(da.coords["channel"].dtype, np.object_)
    assert np.issubdtype(da.coords["longitude"].dtype, np.floating)
    assert np.issubdtype(da.coords["latitude"].dtype, np.floating)

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
    assert np.issubdtype(da.dtype, np.number)
    assert np.issubdtype(da.coords["init_time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["step"].dtype, np.timedelta64)
    assert np.issubdtype(da.coords["channel"].dtype, np.str_)
    assert np.issubdtype(da.coords["x_geostationary"].dtype, np.floating)
    assert np.issubdtype(da.coords["y_geostationary"].dtype, np.floating)

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
