import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.lightarray import LightDataArray
from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.load.nwp import open_nwp
from ocf_data_sampler.load.satellite import open_sat_data


@pytest.fixture()
def xr_gen(generation_zarr_path) -> xr.DataArray:
    return open_generation(generation_zarr_path)

@pytest.fixture()
def xr_ukv(nwp_ukv_zarr_path) -> xr.DataArray:
    return open_nwp(zarr_path=nwp_ukv_zarr_path, provider="ukv")

@pytest.fixture()
def xr_sat(sat_zarr_path) -> xr.DataArray:
    return open_sat_data(zarr_path=sat_zarr_path)


def test_conversion(xr_gen, xr_ukv, xr_sat):
    """Test converting xarray DataArray to LightDataArray and back again"""
    for xda in [xr_gen, xr_ukv, xr_sat]:
        fda = LightDataArray.from_xarray(xda)
        new_xda = fda.to_xarray()
        assert new_xda.equals(xda)


def test_isel(xr_gen, xr_ukv, xr_sat):

    fr_gen = LightDataArray.from_xarray(xr_gen)
    fr_ukv = LightDataArray.from_xarray(xr_ukv)
    fr_sat = LightDataArray.from_xarray(xr_sat)

    index_tasks_1D = [
        # single indexes
        (fr_gen, xr_gen, {"time_utc": 0}),
        (fr_sat, xr_sat, {"time_utc": 13}),
        (fr_gen, xr_gen, {"gen_param": 0}),
        (fr_sat, xr_sat, {"x_geostationary": 13}),
        # slices
        (fr_gen, xr_gen, {"time_utc": slice(0, 10)}),
        (fr_sat, xr_sat, {"time_utc": slice(20, 21)}),
        (fr_gen, xr_gen, {"time_utc": slice(3, None)}),
        # arrays
        (fr_gen, xr_gen, {"location_id": [0]}),
        (fr_gen, xr_gen, {"time_utc": np.array([3,5,7])}),
        (fr_sat, xr_sat, {"time_utc": np.array([5,3,7])}),
    ]
    for fda, xda, indexer in index_tasks_1D:
        sliced_fda = fda.isel(**indexer)
        sliced_xda = xda.isel(**indexer)
        assert sliced_fda.to_xarray().equals(sliced_xda)

    index_tasks_ND = [
        (fr_gen, xr_gen, {"time_utc": 0, "location_id": 3}),
        (fr_gen, xr_gen, {"time_utc": 0, "gen_param": 0}),
        (fr_ukv, xr_ukv, {"init_time_utc": 0, "step": slice(1, 6), "x_osgb": slice(10, 15)}),
    ]
    for fda, xda, indexer in index_tasks_ND:
        sliced_fda = fda.isel(**indexer)
        sliced_xda = xda.isel(**indexer)
        assert sliced_fda.to_xarray().equals(sliced_xda)
