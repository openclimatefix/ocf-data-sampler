from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.xr_tensorstore import open_zarr_paths
from ocf_data_sampler.datasets.pvnet.materialise import block_until_loaded, materialise_data


def _save_zarr(path, values):
    """Save `values` as `dummy_array` in a zarr store."""
    x = np.arange(values.shape[0], dtype=np.float64)
    da = xr.DataArray(values, dims=("x", "y"), coords={"x": x})
    da.to_dataset(name="dummy_array").to_zarr(path)


@pytest.mark.parametrize("source_key", ["sat", "generation"])
def test_block_until_loaded(tmp_path, source_key):
    _save_zarr(tmp_path, np.arange(25, dtype=np.float32).reshape(5, 5))

    # Create a nested dictionary with tensorstore arrays
    source_dict = {
        source_key: open_zarr_paths(str(tmp_path)).dummy_array,
        "nwp": {"ukv": open_zarr_paths(str(tmp_path)).dummy_array},
    }

    block_until_loaded(source_dict)

    # Assert that the result is no longer lazy
    assert isinstance(source_dict[source_key].data, np.ndarray)
    assert isinstance(source_dict["nwp"]["ukv"].data, np.ndarray)


@pytest.mark.parametrize("use_lightarray", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_materialise_data_skips_loaded_arrays(monkeypatch, use_lightarray, dtype):
    source_values = np.array([1.0, 2.0], dtype=dtype)
    da = xr.DataArray(source_values)
    if use_lightarray:
        da = LightDataArray.from_xarray(da)

    read = Mock(side_effect=AssertionError("Already-loaded data must not be read"))
    load = Mock(side_effect=AssertionError("Already-loaded data must not be loaded"))
    monkeypatch.setattr("ocf_data_sampler.datasets.pvnet.materialise.read_tensorstore", read)
    monkeypatch.setattr(LightDataArray, "read", read)
    monkeypatch.setattr(type(da), "load", load)

    source_dict = {"sat": da}
    materialise_data(source_dict)

    assert source_dict["sat"] is da
    assert da.data.dtype == np.float32
    np.testing.assert_array_equal(da.values, [1.0, 2.0])
    # Loaded arrays may be views of the dataset's source arrays, so must not share memory with them
    assert not np.shares_memory(da.data, source_values)


@pytest.mark.parametrize("use_lightarray", [False, True])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_materialise_data(tmp_path, use_lightarray, dtype):
    values = np.arange(25, dtype=dtype).reshape(5, 5) / 3
    _save_zarr(tmp_path, values)

    # Re-open with tensorstore
    da_ts = open_zarr_paths(str(tmp_path)).dummy_array
    nested_da_ts = open_zarr_paths(str(tmp_path)).dummy_array
    if use_lightarray:
        da_ts = LightDataArray.from_xarray(da_ts)
        nested_da_ts = LightDataArray.from_xarray(nested_da_ts)

    # Create a nested dictionary with tensorstore arrays
    source_dict = {
        "sat": da_ts,
        "nwp": {"ukv": nested_da_ts},
    }

    materialise_data(source_dict)

    # Assert that the result is no longer lazy
    assert isinstance(source_dict["sat"].data, np.ndarray)
    assert isinstance(source_dict["nwp"]["ukv"].data, np.ndarray)

    for da in (source_dict["sat"], source_dict["nwp"]["ukv"]):
        assert da.data.dtype == np.float32
        np.testing.assert_array_equal(da.values, values.astype(np.float32))
        assert da["x"].values.dtype == np.float64
