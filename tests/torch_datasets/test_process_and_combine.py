import pytest
import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.numpy_batch.satellite import SatelliteBatchKey
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp
from ocf_data_sampler.utils import minutes

from ocf_data_sampler.torch_datasets.process_and_combine import (
    process_and_combine_datasets,
    merge_dicts,
    fill_nans_in_arrays,
    compute,
)


NWP_FREQ = pd.Timedelta("3h")


@pytest.fixture(scope="module")
def da_sat_like():
    """Create dummy satellite data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(x), len(y))),
        coords=dict(
            time_utc=(["time_utc"], datetimes),
            x_geostationary=(["x_geostationary"], x),
            y_geostationary=(["y_geostationary"], y),
        )
    )
    return da_sat


@pytest.fixture(scope="module")
def da_nwp_like():
    """Create dummy  NWP data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq=NWP_FREQ)
    steps = pd.timedelta_range("0h", "16h", freq="1h")
    channels = ["t", "dswrf"]

    da_nwp = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(steps), len(channels), len(x), len(y))),
        coords=dict(
            init_time_utc=(["init_time_utc"], datetimes),
            step=(["step"], steps),
            channel=(["channel"], channels),
            x_osgb=(["x_osgb"], x),
            y_osgb=(["y_osgb"], y),
        )
    )
    return da_nwp


@pytest.fixture
def mock_constants(monkeypatch):
    """Creation of dummy constants used in normalisation process"""
    mock_nwp_means = {"ukv": {
        "t": 10.0,
        "dswrf": 50.0
    }}
    mock_nwp_stds = {"ukv": {
        "t": 2.0,
        "dswrf": 10.0
    }}
    mock_sat_means = 100.0
    mock_sat_stds = 20.0
    
    monkeypatch.setattr("ocf_data_sampler.constants.NWP_MEANS", mock_nwp_means)
    monkeypatch.setattr("ocf_data_sampler.constants.NWP_STDS", mock_nwp_stds)
    monkeypatch.setattr("ocf_data_sampler.constants.SAT_MEANS", mock_sat_means)
    monkeypatch.setattr("ocf_data_sampler.constants.SAT_STDS", mock_sat_stds)


@pytest.fixture
def mock_config():
    """Specify dummy configuration"""
    class MockConfig:
        class InputData:
            class NWP:
                provider = "ukv"
                interval_start_minutes = -360
                interval_end_minutes = 180
                time_resolution_minutes = 60

            class GSP:
                interval_start_minutes = -120
                interval_end_minutes = 120
                time_resolution_minutes = 30

            def __init__(self):
                self.nwp = {"ukv": self.NWP()}
                self.gsp = self.GSP()

        def __init__(self):
            self.input_data = self.InputData()
    
    return MockConfig()


@pytest.fixture
def mock_location():
    """Create dummy location"""
    return Location(id=12345, x=400000, y=500000)


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
    """Test the compute function"""
    da = xr.DataArray(np.random.rand(5, 5))

    # Create nested dictionary
    nested_dict = {
        "array1": da,
        "nested": {
            "array2": da
        }
    }
    
    result = compute(nested_dict)
    
    # Ensure function applied - check if data is no longer lazy array and determine structural alterations
    # Check that result is an xarray DataArray
    assert isinstance(result["array1"], xr.DataArray)
    assert isinstance(result["nested"]["array2"], xr.DataArray)
    
    # Check data is no longer lazy object
    assert isinstance(result["array1"].data, np.ndarray)
    assert isinstance(result["nested"]["array2"].data, np.ndarray)
    
    # Check for NaN
    assert not np.isnan(result["array1"].data).any()
    assert not np.isnan(result["nested"]["array2"].data).any()


# TO DO - Update the below to include satellite and finalise testing procedure
# Currently for NWP only - awaiting confirmation
@pytest.mark.parametrize("t0_str", ["10:00", "11:00", "12:00"])
def test_full_pipeline(da_nwp_like, mock_config, mock_location, mock_constants, t0_str):
    """Test full pipeline considering time slice selection and then process and combine"""
    t0 = pd.Timestamp(f"2024-01-02 {t0_str}")
    
    # Obtain NWP data slice
    nwp_sample = select_time_slice_nwp(
        da_nwp_like,
        t0,
        sample_period_duration=pd.Timedelta(minutes=mock_config.input_data.nwp["ukv"].time_resolution_minutes),
        interval_start=pd.Timedelta(minutes=mock_config.input_data.nwp["ukv"].interval_start_minutes),
        interval_end=pd.Timedelta(minutes=mock_config.input_data.nwp["ukv"].interval_end_minutes),
        dropout_timedeltas=None,
        dropout_frac=0,
        accum_channels=["dswrf"],
        channel_dim_name="channel",
    )
    
    # Prepare dataset dictionary
    dataset_dict = {
        "nwp": {"ukv": nwp_sample},
    }
    
    # Process data with main function
    result = process_and_combine_datasets(
        dataset_dict,
        mock_config,
        t0,
        mock_location,
        target_key='gsp'
    )
    
    # Verify results structure
    assert NWPBatchKey.nwp in result
    
    # Check NWP data normalisation and NaN handling
    nwp_data = result[NWPBatchKey.nwp]["ukv"]
    assert isinstance(nwp_data['nwp'], np.ndarray)
    assert not np.isnan(nwp_data['nwp']).any()
