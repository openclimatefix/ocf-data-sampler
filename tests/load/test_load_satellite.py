import numpy as np
import xarray as xr

from ocf_data_sampler.load.satellite import open_sat_data


def test_open_satellite(sat_zarr_path):
    da = open_sat_data(zarr_path=sat_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "channel", "x_geostationary", "y_geostationary")
    # 288 is 1 days of data at 5 minutes intervals, 12 * 24
    # There are 11 channels
    # There are 100 x 100 pixels
    assert da.shape == (288, 11, 100, 100)
    assert np.issubdtype(da.dtype, np.number)
