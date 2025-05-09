import numpy as np
from xarray import DataArray

from ocf_data_sampler.config.model import NWP
from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path, nwp_ukv_config):
    nwp_ukv_config.zarr_path = nwp_ukv_zarr_path

    da = open_nwp(nwp_config=nwp_ukv_config)
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_ecmwf(nwp_ecmwf_zarr_path, nwp_ukv_config):
    nwp_ukv_config.zarr_path = nwp_ecmwf_zarr_path
    nwp_ukv_config.provider = "ecmwf"

    da = open_nwp(nwp_ukv_config)
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert np.issubdtype(da.dtype, np.number)


def test_load_icon_eu(icon_eu_zarr_path, nwp_ukv_config):
    nwp_ukv_config.zarr_path = icon_eu_zarr_path
    nwp_ukv_config.provider = "icon-eu"

    da = open_nwp(nwp_ukv_config)
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (2, 78, 2, 100, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_cloudcasting(nwp_cloudcasting_zarr_path, nwp_ukv_config):
    nwp_ukv_config.zarr_path = nwp_cloudcasting_zarr_path
    nwp_ukv_config.provider = "cloudcasting"

    da = open_nwp(nwp_ukv_config)
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")
    assert "area" in da.attrs
    assert da.shape == (2, 12, 3, 100, 100)
    assert np.issubdtype(da.dtype, np.number)
