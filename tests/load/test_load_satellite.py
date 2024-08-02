from ocf_data_sampler.load.satellite import open_sat_data
import xarray as xr
import numpy as np


def test_open_satellite(sat_zarr_path):
    da = open_sat_data(zarr_path=sat_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "y_geostationary", "x_geostationary")
    assert da.shape == (576, 11, 20, 49)
    assert da.dtype == np.float32


