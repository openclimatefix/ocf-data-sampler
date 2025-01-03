import pytest
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.numpy_batch import NWPBatchKey, GSPBatchKey, SatelliteBatchKey
from ocf_data_sampler.torch_datasets import PVNetUKRegionalDataset

from ocf_data_sampler.torch_datasets.process_and_combine import (
    process_and_combine_datasets,
    process_and_combine_site_sample_dict,
    merge_dicts,
    fill_nans_in_arrays,
    compute,
)


def test_process_and_combine_datasets(pvnet_config_filename):

    # Load in config for function and define location 
    config = load_yaml_configuration(pvnet_config_filename)
    t0 = pd.Timestamp("2024-01-01 00:00")
    location = Location(coordinate_system="osgb", x=1234, y=5678, id=1)

    nwp_data = xr.DataArray(
        np.random.rand(4, 2, 2, 2),
        dims=["time_utc", "channel", "y", "x"],
        coords={
            "time_utc": pd.date_range("2024-01-01 00:00", periods=4, freq="h"),
            "channel": ["t2m", "dswrf"],
            "step": ("time_utc", pd.timedelta_range(start='0h', periods=4, freq='h')),
            "init_time_utc": pd.Timestamp("2024-01-01 00:00")
        }
    )

    sat_data = xr.DataArray(
        np.random.rand(7, 1, 2, 2),
        dims=["time_utc", "channel", "y", "x"],
        coords={
            "time_utc": pd.date_range("2024-01-01 00:00", periods=7, freq="5min"),
            "channel": ["HRV"],
            "x_geostationary": (["y", "x"], np.array([[1, 2], [1, 2]])),
            "y_geostationary": (["y", "x"], np.array([[1, 1], [2, 2]]))
        }
    )

    # Combine as dict
    dataset_dict = {
        "nwp": {"ukv": nwp_data},
        "sat": sat_data
    }

    # Call relevant function
    result = process_and_combine_datasets(dataset_dict, config, t0, location)

    # Assert result is dict - check and validate
    assert isinstance(result, dict)
    assert NWPBatchKey.nwp in result
    assert result[SatelliteBatchKey.satellite_actual].shape == (7, 1, 2, 2)
    assert result[NWPBatchKey.nwp]["ukv"][NWPBatchKey.nwp].shape == (4, 1, 2, 2)


def test_merge_dicts():
    """Test merge_dicts function"""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    dict3 = {"e": 5}
    
    result = merge_dicts([dict1, dict2, dict3])
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    
    # Test key overwriting
    dict4 = {"a": 10, "f": 6}
    result = merge_dicts([dict1, dict4])
    assert result["a"] == 10


def test_fill_nans_in_arrays():
    """Test the fill_nans_in_arrays function"""
    array_with_nans = np.array([1.0, np.nan, 3.0, np.nan])
    nested_dict = {
        "array1": array_with_nans,
        "nested": {
            "array2": np.array([np.nan, 2.0, np.nan, 4.0])
        },
        "string_key": "not_an_array"
    }
    
    result = fill_nans_in_arrays(nested_dict)
    
    assert not np.isnan(result["array1"]).any()
    assert np.array_equal(result["array1"], np.array([1.0, 0.0, 3.0, 0.0]))
    assert not np.isnan(result["nested"]["array2"]).any()
    assert np.array_equal(result["nested"]["array2"], np.array([0.0, 2.0, 0.0, 4.0]))
    assert result["string_key"] == "not_an_array"


def test_compute():
    """Test compute function with dask array"""
    da_dask = xr.DataArray(da.random.random((5, 5)))

    # Create a nested dictionary with dask array
    nested_dict = {
        "array1": da_dask,
        "nested": {
            "array2": da_dask
        }
    }

    # Ensure initial data is lazy - i.e. not yet computed
    assert not isinstance(nested_dict["array1"].data, np.ndarray)
    assert not isinstance(nested_dict["nested"]["array2"].data, np.ndarray)

    # Call the compute function
    result = compute(nested_dict)

    # Assert that the result is an xarray DataArray and no longer lazy
    assert isinstance(result["array1"], xr.DataArray)
    assert isinstance(result["nested"]["array2"], xr.DataArray)
    assert isinstance(result["array1"].data, np.ndarray)
    assert isinstance(result["nested"]["array2"].data, np.ndarray)

    # Ensure there no NaN values in computed data
    assert not np.isnan(result["array1"].data).any()
    assert not np.isnan(result["nested"]["array2"].data).any()


def test_process_and_combine_site_sample_dict(pvnet_config_filename):
    # Load config
    config = load_yaml_configuration(pvnet_config_filename)

    # Specify minimal structure for testing
    raw_nwp_values = np.random.rand(4, 1, 2, 2)  # Single channel
    site_dict = {
        "nwp": {
            "ukv": xr.DataArray(
                raw_nwp_values,
                dims=["time_utc", "channel", "y", "x"],
                coords={
                    "time_utc": pd.date_range("2024-01-01 00:00", periods=4, freq="h"),
                    "channel": ["dswrf"],  # Single channel
                },
            )
        }
    }
    print(f"Input site_dict: {site_dict}")

    # Call function
    result = process_and_combine_site_sample_dict(site_dict, config)

    # Assert to validate output structure
    assert isinstance(result, xr.Dataset), "Result should be an xarray.Dataset"
    assert len(result.data_vars) > 0, "Dataset should contain data variables"

    # Validate variable via assertion and shape of such
    expected_variable = "nwp-ukv"
    assert expected_variable in result.data_vars, f"Expected variable '{expected_variable}' not found"
    nwp_result = result[expected_variable]
    assert nwp_result.shape == (4, 1, 2, 2), f"Unexpected shape for '{expected_variable}': {nwp_result.shape}"
