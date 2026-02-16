from ocf_data_sampler.torch_datasets.fastarray import FastDataArray

from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.load.nwp import open_nwp
from ocf_data_sampler.load.satellite import open_sat_data

import pytest
import xarray as xr
import numpy as np

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
    """Test converting xarray DataArray to FastDataArray and back again"""
    for xda in [xr_gen, xr_ukv, xr_sat]:
        fda = FastDataArray.from_xarray(xda)
        new_xda = fda.to_xarray()
        assert new_xda.equals(xda)


def test_isel(xr_gen, xr_ukv, xr_sat):

    fr_gen = FastDataArray.from_xarray(xr_gen)
    fr_ukv = FastDataArray.from_xarray(xr_ukv)
    fr_sat = FastDataArray.from_xarray(xr_sat)

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


def test_sel(xr_gen, xr_ukv, xr_sat):

    fr_gen = FastDataArray.from_xarray(xr_gen)
    fr_ukv = FastDataArray.from_xarray(xr_ukv)
    fr_sat = FastDataArray.from_xarray(xr_sat)


    gen_times = xr_gen.time_utc.values
    sat_times = xr_sat.time_utc.values
    index_tasks_1D = [
        # single indexes
        (fr_gen, xr_gen, {"time_utc": gen_times[4]}),
        (fr_sat, xr_sat, {"time_utc": gen_times[-2]}),
        (fr_sat, xr_sat, {"time_utc": sat_times[-1]}),
        (fr_gen, xr_gen, {"gen_param": "generation_mw"}),
        (fr_sat, xr_sat, {"channel": "VIS008"}),
        # slices
        (fr_gen, xr_gen, {"time_utc": slice(gen_times[3], gen_times[10])}), 
        (fr_gen, xr_gen, {"location_id": slice(2, 5)}),
        (fr_gen, xr_gen, {"location_id": slice(2, None)}),
        # arrays
        (fr_gen, xr_gen, {"time_utc": gen_times[3:8]}),
        (fr_sat, xr_sat, {"channel": ["VIS008", "IR_134"]}),
    ]
    for fda, xda, indexer in index_tasks_1D:
        sliced_fda = fda.sel(**indexer)
        sliced_xda = xda.sel(**indexer)
        assert sliced_fda.to_xarray().equals(sliced_xda)

    ukv_times = xr_ukv.init_time_utc.values
    ukv_steps = xr_ukv.step.values
    ukv_x = xr_ukv.x_osgb.values
    index_tasks_ND = [
        (fr_gen, xr_gen, {"time_utc": gen_times[0], "location_id": 3}),
        (fr_gen, xr_gen, {"time_utc": gen_times[0], "gen_param": "generation_mw"}),
        (fr_ukv, xr_ukv, {
            "init_time_utc": ukv_times[0], 
            "step": slice(ukv_steps[1], ukv_steps[6]), 
            "x_osgb": slice(ukv_x[10], ukv_x[15])
        }),
    ]
    for fda, xda, indexer in index_tasks_ND:
        sliced_fda = fda.sel(**indexer)
        sliced_xda = xda.sel(**indexer)
        assert sliced_fda.to_xarray().equals(sliced_xda)

