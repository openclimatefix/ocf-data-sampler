import pandas as pd
from xarray import DataArray
import numpy as np

from ocf_data_sampler.load.nwp import open_nwp


def test_load_ukv(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert np.issubdtype(da.dtype, np.number)


def test_load_ecmwf(nwp_ecmwf_zarr_path):
    da = open_nwp(zarr_path=nwp_ecmwf_zarr_path, provider="ecmwf")
    assert isinstance(da, DataArray)
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert np.issubdtype(da.dtype, np.number)