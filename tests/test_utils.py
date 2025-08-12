import dask.array
import numpy as np
import xarray as xr

from ocf_data_sampler.load.open_xarray_tensorstore import open_zarr
from ocf_data_sampler.utils import compute, tensorstore_compute


def test_compute():
    """Test compute function with dask array"""
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))

    # Create a nested dictionary with dask array
    lazy_data_dict = {
        "array1": da_dask,
        "nested": {"array2": da_dask},
    }

    computed_data_dict = compute(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(computed_data_dict["array1"].data, np.ndarray)
    assert isinstance(computed_data_dict["nested"]["array2"].data, np.ndarray)


def test_tensorstore_compute(tmp_path):

    # Save a zarr
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))
    da_dask.to_dataset(name="array").to_zarr(tmp_path)

    # Re-open with tensorstore
    da_ts = open_zarr(tmp_path).array

    # Create a nested dictionary with tensorstore arrays
    lazy_data_dict = {
        "array1": da_ts,
        "nested": {"array2": da_ts},
    }

    computed_data_dict = tensorstore_compute(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(computed_data_dict["array1"].data, np.ndarray)
    assert isinstance(computed_data_dict["nested"]["array2"].data, np.ndarray)
