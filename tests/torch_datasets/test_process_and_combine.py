import pytest
import numpy as np
import pandas as pd
import xarray as xr

from unittest.mock import MagicMock

from ocf_data_sampler.constants import SAT_MEANS, SAT_STDS, NWP_MEANS, NWP_STDS, EPSILON
from ocf_data_sampler.numpy_batch.satellite import SatelliteBatchKey
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.torch_datasets.process_and_combine import (
    process_and_combine_datasets,
    merge_dicts,
    fill_nans_in_arrays,
    compute,
)


@pytest.fixture(scope="module")
def da_sat_like():
    """ Create dummy satellite DataArray """
    data = np.random.rand(3, 4, 4).astype(np.float32)
    coords = {
        "time_utc": pd.date_range("2023-01-01", periods=3),
        "x_geostationary": np.arange(4),
        "y_geostationary": np.arange(4),
    }
    return xr.DataArray(
        data,
        coords=coords,
        dims=["time_utc", "y_geostationary", "x_geostationary"],
    )


@pytest.fixture(scope="module")
def da_nwp_like():
    """ Create dummy NWP DataArray """
    data = np.random.rand(3, 3, 4, 4).astype(np.float32)
    coords = {
        "init_time_utc": pd.date_range("2023-01-01", periods=3),
        "step": pd.timedelta_range("0h", periods=3, freq="1h"),
        "x_osgb": np.arange(4),
        "y_osgb": np.arange(4),
    }
    return xr.DataArray(
        data,
        coords=coords,
        dims=["init_time_utc", "step", "y_osgb", "x_osgb"],
    )


def test_merge_dicts():
    dicts = [
        {"a": 1, "b": 2},
        {"c": 3, "d": 4},
        {"b": 5, "e": 6},
    ]
    result = merge_dicts(dicts)
    expected = {"a": 1, "b": 5, "c": 3, "d": 4, "e": 6}
    assert result == expected


def test_fill_nans_in_arrays():
    batch = {
        "array_1": np.array([1, np.nan, 3]),
        "array_2": np.array([[np.nan, 2], [3, np.nan]]),
        "nested_dict": {
            "array_3": np.array([np.nan, 0]),
        },
        "non_numeric": "keep_this",
    }

    result = fill_nans_in_arrays(batch)
    expected = {
        "array_1": np.array([1, 0, 3]),
        "array_2": np.array([[0, 2], [3, 0]]),
        "nested_dict": {
            "array_3": np.array([0, 0]),
        },
        "non_numeric": "keep_this",
    }

    np.testing.assert_array_equal(result["array_1"], expected["array_1"])
    np.testing.assert_array_equal(result["array_2"], expected["array_2"])
    np.testing.assert_array_equal(result["nested_dict"]["array_3"], expected["nested_dict"]["array_3"])
    assert result["non_numeric"] == expected["non_numeric"]


def test_compute():
    mock_dataarray = MagicMock()
    mock_dataarray.compute = MagicMock(side_effect=lambda scheduler=None: mock_dataarray)

    xarray_dict = {
        "level1": {
            "level2": mock_dataarray,
        },
        "another_level1": mock_dataarray,
    }

    result = compute(xarray_dict)
    assert result["level1"]["level2"] == mock_dataarray
    assert result["another_level1"] == mock_dataarray
    mock_dataarray.compute.assert_called()


def test_process_and_combine_datasets(da_sat_like, da_nwp_like):
    # Dummy config with valid integer values
    mock_config = MagicMock()
    mock_config.input_data.nwp = {"ukv": MagicMock(provider="ukv")}
    mock_config.input_data.satellite = {"rss": MagicMock(provider="rss")}
    mock_config.input_data.gsp = MagicMock(
        interval_start_minutes=-30,
        interval_end_minutes=30,
        time_resolution_minutes=15,
    )

    t0 = pd.Timestamp("2023-01-01 00:00:00")
    mock_location = MagicMock(x=12345.6, y=65432.1, id=1)

    dataset_dict = {
        "nwp": {"ukv": da_nwp_like},
        "sat": {"rss": da_sat_like},
    }

    # Run function
    result = process_and_combine_datasets(dataset_dict, mock_config, t0, mock_location)

    # Assertion currently only for sattelite and NWP
    assert isinstance(result, dict)
    assert SatelliteBatchKey.satellite_actual in result
    assert NWPBatchKey.nwp in result

    # Assert no NaNs remain
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            assert not np.isnan(value).any()
