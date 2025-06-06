import numpy as np
import pandas as pd
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

    assert np.issubdtype(da.coords["time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["channel"].dtype, np.str_)
    assert np.issubdtype(da.coords["x_geostationary"].dtype, np.floating)
    assert np.issubdtype(da.coords["y_geostationary"].dtype, np.floating)

    # Permissibility for NaN in Sat
    assert not da.coords["time_utc"].isnull().any()
    assert not da.coords["channel"].isnull().any()
    assert not da.coords["x_geostationary"].isnull().any()
    assert not da.coords["y_geostationary"].isnull().any()

    expected_freq = pd.to_timedelta("5 minutes")
    time_diffs = da.coords["time_utc"].diff("time_utc")
    if len(time_diffs) > 0:
        assert (time_diffs == expected_freq).all()

    assert len(np.unique(da.coords["channel"])) == da.shape[1]
