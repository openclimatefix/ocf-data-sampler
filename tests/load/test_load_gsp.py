from ocf_data_sampler.load.gsp import open_gsp
import xarray as xr


def test_open_gsp(uk_gsp_zarr_path):
    da = open_gsp(uk_gsp_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "gsp_id")

    assert "nominal_capacity_mwp" in da.coords
    assert "effective_capacity_mwp" in da.coords
    assert "x_osgb" in da.coords
    assert "y_osgb" in da.coords
    assert da.shape == (49, 318)
