import xarray as xr
import numpy as np

from ocf_data_sampler.load.site import open_site


def test_open_site(data_sites):
    da = open_site(data_sites.file_path, data_sites.metadata_file_path, data_sites.capacity_mode)

    assert da.generation_kw.dims == ("time_utc", "site_id")
    assert "capacity_kwp" in da.coords
    assert "latitude" in da.coords
    assert "longitude" in da.coords
    assert da.generation_kw.shape == (49, 10)
    # Check that capacity_kwp is a 1D coordinate with site_id dimension
    assert da.coords["capacity_kwp"].dims == ("site_id",)

