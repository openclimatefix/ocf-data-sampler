import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.load.site import open_site


def test_open_site(default_data_site_model):
    da = open_site(default_data_site_model.file_path, default_data_site_model.metadata_file_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "site_id")
    assert "capacity_kwp" in da.coords
    assert "latitude" in da.coords
    assert "longitude" in da.coords
    assert da.shape == (49, 10)

    assert np.issubdtype(da.dtype, np.floating)

    assert np.issubdtype(da.coords["time_utc"].dtype, np.datetime64)
    assert np.issubdtype(da.coords["site_id"].dtype, np.integer)
    assert np.issubdtype(da.coords["capacity_kwp"].dtype, np.floating)
    assert np.issubdtype(da.coords["latitude"].dtype, np.floating)
    assert np.issubdtype(da.coords["longitude"].dtype, np.floating)

    assert not da.isnull().any()
    assert not da.coords["capacity_kwp"].isnull().any()
    assert not da.coords["latitude"].isnull().any()
    assert not da.coords["longitude"].isnull().any()

    expected_freq = pd.to_timedelta("30 minutes")
    time_diffs = da.coords["time_utc"].diff("time_utc")
    if len(time_diffs) > 0:
        assert (time_diffs == expected_freq).all()

    assert len(np.unique(da.coords["site_id"])) == da.shape[1]
