import numpy as np
import pytest
import tensorstore as ts
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
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
        (fr_gen, xr_gen, {"gen_param": np.array([True, False])}),
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


@pytest.mark.parametrize(
    "indexers",
    [
        {"a": [0, 1], "b": [0, 2]},
        {"a": 0, "c": [0, 2]},
        {"a": 1, "b": slice(1, None), "c": [0, 2], "d": 3},
        {"a": [True, False], "b": np.array([False, True, True])},
        {"a": np.array([False, True]), "c": 2, "d": [True, False, True, False, True]},
    ],
)
def test_isel_conforms_to_xarray(indexers):
    """Test that multiple indexers preserve xarray's dimension order and values."""
    dims = ("a", "b", "c", "d")
    shape = (2, 3, 4, 5)
    xda = xr.DataArray(
        np.arange(np.prod(shape)).reshape(shape),
        dims=dims,
        coords={dim: np.arange(size) for dim, size in zip(dims, shape, strict=True)},
    )
    fda = LightDataArray.from_xarray(xda)

    sliced_fda = fda.isel(**indexers)
    sliced_xda = xda.isel(**indexers)

    assert sliced_fda.to_xarray().equals(sliced_xda)


def test_isel_rejects_boolean_mask_with_wrong_length():
    """Test that a boolean mask must match the length of its indexed dimension."""
    xda = xr.DataArray(np.arange(6).reshape(2, 3), dims=("a", "b"))
    fda = LightDataArray.from_xarray(xda)

    with pytest.raises(IndexError, match="dimension has length 2"):
        fda.isel(a=[True])


def test_isel_tensorstore_slices_are_relative_to_current_array():
    """Test that repeated TensorStore slicing uses NumPy-style relative indices."""
    xda = xr.DataArray(np.arange(20), dims=("x",), coords={"x": np.arange(20)})
    fda = LightDataArray(
        data=ts.array(xda.values),
        dims=("x",),
        coords={"x": (("x",), xda["x"].values)},
    )

    sliced_fda = fda.isel(x=slice(10, 20)).isel(x=slice(0, 3))
    sliced_xda = xda.isel(x=slice(10, 20)).isel(x=slice(0, 3))

    assert sliced_fda.to_xarray().equals(sliced_xda)
