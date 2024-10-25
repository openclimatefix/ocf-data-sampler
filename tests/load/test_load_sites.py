from ocf_data_sampler.load.sites import open_sites
import xarray as xr


def test_open_site(data_sites):
    da = open_sites(data_sites)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "system_id")

    assert "capacity_kwp" in da.coords
    assert "latitude" in da.coords
    assert "longitude" in da.coords
    assert da.shape == (49, 10)
