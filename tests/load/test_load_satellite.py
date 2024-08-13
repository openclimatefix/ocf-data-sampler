from ocf_data_sampler.load.satellite import open_sat_data
import xarray as xr
import numpy as np


def test_open_satellite(sat_zarr_path):
    da = open_sat_data(zarr_path=sat_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "x_geostationary", "y_geostationary")
    # 576 is 2 days of data at 5 minutes intervals, 12 * 24 * 2
    # There are 11 channels
    # There are 49 x 20 pixels
    assert da.shape == (576, 11, 49, 20)
    assert np.issubdtype(da.dtype, np.number)


