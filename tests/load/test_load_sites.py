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
