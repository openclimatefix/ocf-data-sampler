import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.load.gsp import get_gsp_boundaries, open_gsp


@pytest.mark.parametrize("version, expected_length", [("20220314", 318), ("20250109", 332)])
def test_get_gsp_boundaries(version, expected_length):
    df = get_gsp_boundaries(version)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == expected_length
    assert "x_osgb" in df.columns
    assert "y_osgb" in df.columns

    assert df.index.is_unique


def test_open_gsp(uk_gsp_zarr_path):
    da = open_gsp(uk_gsp_zarr_path)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time_utc", "gsp_id")

    assert "nominal_capacity_mwp" in da.coords
    assert "effective_capacity_mwp" in da.coords
    assert "x_osgb" in da.coords
    assert "y_osgb" in da.coords
    assert da.shape == (49, 318)


