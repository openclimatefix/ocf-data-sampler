import dask.array
import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.xr_tensorstore import open_zarr_paths
from ocf_data_sampler.datasets.pvnet.materialise import block_until_loaded, materialise_data


def test_block_until_loaded():
    """Test load function with dask array"""
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))

    # Create a nested dictionary with dask array
    lazy_data_dict = {
        "array1": da_dask,
        "nested": {"array2": da_dask},
    }

    loaded_data_dict = block_until_loaded(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(loaded_data_dict["array1"].data, np.ndarray)
    assert isinstance(loaded_data_dict["nested"]["array2"].data, np.ndarray)


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
    lazy_data_dict = {
        "array1": da_ts,
        "nested": {"array2": nested_da_ts},
    }

    loaded_data_dict = materialise_data(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(loaded_data_dict["array1"].data, np.ndarray)
    assert isinstance(loaded_data_dict["nested"]["array2"].data, np.ndarray)

    for da in (loaded_data_dict["array1"], loaded_data_dict["nested"]["array2"]):
        assert da.data.dtype == np.float32
        np.testing.assert_array_equal(da.values, values.astype(np.float32))
        assert da["x"].values.dtype == np.float64
