from unittest.mock import Mock

import dask.array
import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.xr_tensorstore import open_zarr_paths
from ocf_data_sampler.datasets.pvnet.materialise import block_until_loaded, materialise_data


@pytest.mark.parametrize("source_key", ["array1", "generation"])
def test_block_until_loaded(source_key):
    """Test load function with dask array"""
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))

    # Create a nested dictionary with dask arrays
    source_dict = {
        source_key: da_dask,
        "nested": {"array2": da_dask.copy()},
    }

    block_until_loaded(source_dict)

    # Assert that the result is no longer lazy
    assert isinstance(source_dict[source_key].data, np.ndarray)
    assert isinstance(source_dict["nested"]["array2"].data, np.ndarray)


@pytest.mark.parametrize("use_lightarray", [False, True])
def test_materialise_data_skips_loaded_arrays(monkeypatch, use_lightarray):
    da = xr.DataArray(np.array([1.0, 2.0], dtype=np.float64))
    if use_lightarray:
        da = LightDataArray.from_xarray(da)

    read = Mock(side_effect=AssertionError("Already-loaded data must not be read"))
    load = Mock(side_effect=AssertionError("Already-loaded data must not be loaded"))
    monkeypatch.setattr("ocf_data_sampler.datasets.pvnet.materialise.xtr_read", read)
    monkeypatch.setattr(LightDataArray, "read", read)
    monkeypatch.setattr(type(da), "load", load)

    source_dict = {"sat": da}
    materialise_data(source_dict)

    assert source_dict["sat"] is da
    assert da.data.dtype == np.float32
    np.testing.assert_array_equal(da.values, [1.0, 2.0])


@pytest.mark.parametrize("use_lightarray", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_materialise_data(tmp_path, use_lightarray, dtype):

    # Save a zarr
    values = np.arange(25, dtype=dtype).reshape(5, 5) / 3
    da_dask = xr.DataArray(values, dims=("x", "y"), coords={"x": np.arange(5, dtype=np.float64)})
    da_dask.to_dataset(name="dummy_array").to_zarr(tmp_path)

    # Re-open with tensorstore
    da_ts = open_zarr_paths(str(tmp_path)).dummy_array
    nested_da_ts = open_zarr_paths(str(tmp_path)).dummy_array
    if use_lightarray:
        da_ts = LightDataArray.from_xarray(da_ts)
        nested_da_ts = LightDataArray.from_xarray(nested_da_ts)

    # Create a nested dictionary with tensorstore arrays
    source_dict = {
        "array1": da_ts,
        "nested": {"array2": nested_da_ts},
    }

    materialise_data(source_dict)

    # Assert that the result is no longer lazy
    assert isinstance(source_dict["array1"].data, np.ndarray)
    assert isinstance(source_dict["nested"]["array2"].data, np.ndarray)

    for da in (source_dict["array1"], source_dict["nested"]["array2"]):
        assert da.data.dtype == np.float32
        np.testing.assert_array_equal(da.values, values.astype(np.float32))
        assert da["x"].values.dtype == np.float64
