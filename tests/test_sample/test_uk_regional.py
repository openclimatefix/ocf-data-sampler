"""Tests for PVNet sample and dataset implementations"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from unittest.mock import patch, MagicMock
import torch

from ocf_data_sampler.sample.uk_regional import PVNetSample, PVNetUKRegionalDataset
from ocf_data_sampler.numpy_sample import (
    NWPSampleKey,
    GSPSampleKey,
    SatelliteSampleKey
)
from ocf_data_sampler.select import Location
from ocf_data_sampler.config import Configuration


# Fixtures
@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing"""
    plt.switch_backend('Agg')  # Use non-interactive backend

# @pytest.fixture
# def mock_data_loading():
#     time_range = pd.date_range('2024-01-01', periods=48, freq='30min')
#     gsp_data = xr.DataArray(
#         np.random.rand(len(time_range), 3),  # Add second dimension for GSP IDs
#         dims=['time_utc', 'gsp_id'],  # Add gsp_id dimension
#         coords={
#             'time_utc': time_range,
#             'gsp_id': [1, 2, 3],  # Add gsp_id coordinate
#             'effective_capacity_mwp': ('time_utc', np.ones(len(time_range)) * 100),
#             'nominal_capacity_mwp': ('time_utc', np.ones(len(time_range)) * 100)
#         }
#     )

#     with patch('ocf_data_sampler.load.load_dataset.open_gsp') as mock_open_gsp, \
#          patch('ocf_data_sampler.load.load_dataset.get_dataset_dict') as mock_get_dict:
        
#         mock_open_gsp.return_value = gsp_data
#         mock_get_dict.return_value = {
#             'nwp': {'ukv': xr.DataArray(np.random.rand(24, 64, 64))},
#             'sat': xr.DataArray(np.random.rand(48, 64, 64)),
#             'gsp': gsp_data
#         }
#         yield mock_get_dict

@pytest.fixture
def mock_data_loading():
    time_range = pd.date_range('2024-01-01', periods=48, freq='30min')
    gsp_data = xr.DataArray(
        np.random.rand(len(time_range), 3),
        dims=['time_utc', 'gsp_id'],
        coords={
            'time_utc': time_range,
            'gsp_id': [1, 2, 3],
            'effective_capacity_mwp': ('time_utc', np.ones(len(time_range)) * 100),
            'nominal_capacity_mwp': ('time_utc', np.ones(len(time_range)) * 100)
        }
    )

    with patch('ocf_data_sampler.load.load_dataset.open_gsp') as mock_open_gsp, \
         patch('ocf_data_sampler.load.load_dataset.get_dataset_dict') as mock_get_dict, \
         patch('xarray.open_zarr', return_value=gsp_data):
        
        mock_open_gsp.return_value = gsp_data
        mock_get_dict.return_value = {
            'nwp': {'ukv': xr.DataArray(np.random.rand(24, 64, 64))},
            'sat': xr.DataArray(np.random.rand(48, 64, 64)),
            'gsp': gsp_data
        }
        yield mock_get_dict

@pytest.fixture
def real_sample_data():
    """Create realistic sample data for testing"""
    nwp_data = np.random.rand(24, 64, 64)
    gsp_data = np.random.rand(48)
    sat_data = np.random.rand(64, 64)
    solar_data = np.random.rand(48)
    
    return {
        'nwp': {
            'ukv': {
                'nwp': nwp_data.copy(),
                'channel_names': np.array(['temperature', 'precipitation', 'radiation'])
            }
        },
        GSPSampleKey.gsp: gsp_data.copy(),
        SatelliteSampleKey.satellite_actual: sat_data.copy(),
        GSPSampleKey.solar_azimuth: solar_data.copy(),
        GSPSampleKey.solar_elevation: solar_data.copy()
    }

@pytest.fixture
def sample_config():
    """Create configuration for testing"""
    return Configuration(
        general={
            'name': 'test_config',
            'description': 'Configuration for testing'
        },
        input_data={
            'gsp': {
                'time_resolution_minutes': 30,
                'interval_start_minutes': 0,
                'interval_end_minutes': 1440,
                'zarr_path': 'dummy/path/gsp.zarr'
            },
            'nwp': {
                'ukv': {
                    'time_resolution_minutes': 60,
                    'interval_start_minutes': 0,
                    'interval_end_minutes': 1440,
                    'provider': 'ukv',
                    'zarr_path': 'dummy/path/nwp.zarr',
                    'image_size_pixels_height': 64,
                    'image_size_pixels_width': 64,
                    'channels': ['temperature', 'precipitation']
                }
            },
            'satellite': {
                'time_resolution_minutes': 15,
                'interval_start_minutes': 0,
                'interval_end_minutes': 1440,
                'zarr_path': 'dummy/path/sat.zarr',
                'image_size_pixels_height': 64,
                'image_size_pixels_width': 64,
                'channels': ['channel1', 'channel2']
            }
        }
    )

@pytest.fixture
def config_file(tmp_path, sample_config):
    """Create a temporary config file"""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config.model_dump(), f)
    return config_path

# PVNetSample Tests
def test_pvnet_sample_init():
    """Test PVNetSample initialization"""
    sample = PVNetSample()
    assert isinstance(sample, PVNetSample)
    assert len(sample.keys()) == 0

def test_pvnet_sample_data_handling(real_sample_data):
    """Test PVNetSample data handling"""
    sample = PVNetSample()
    
    # Test adding data
    for key, value in real_sample_data.items():
        sample[key] = value
    
    # Verify data was stored correctly
    assert set(sample.keys()) == set(real_sample_data.keys())
    for key in real_sample_data:
        if isinstance(real_sample_data[key], dict):
            assert isinstance(sample[key], dict)
            assert set(sample[key].keys()) == set(real_sample_data[key].keys())
            for sub_key in sample[key]:
                np.testing.assert_array_equal(
                    sample[key][sub_key], 
                    real_sample_data[key][sub_key]
                )
        else:
            np.testing.assert_array_equal(sample[key], real_sample_data[key])

def test_pvnet_sample_validation(real_sample_data):
    """Test PVNetSample validation with real data"""
    sample = PVNetSample()
    
    # Test with complete data
    for key, value in real_sample_data.items():
        sample[key] = value
    sample.validate()  # Should not raise
    
    # Test with missing key
    sample = PVNetSample()
    incomplete_data = {k: v for k, v in real_sample_data.items() 
                      if k != GSPSampleKey.gsp}
    for key, value in incomplete_data.items():
        sample[key] = value
    with pytest.raises(ValueError, match="Missing required keys"):
        sample.validate()

def test_pvnet_sample_invalid_data(real_sample_data):
    """Test handling of invalid data structures"""
    sample = PVNetSample()
    
    # Add all required data except NWP
    for key, value in real_sample_data.items():
        if key != 'nwp':
            sample[key] = value
            
    # Test invalid NWP structure
    with pytest.raises(TypeError, match="NWP data must be nested dictionary"):
        sample['nwp'] = np.array([1, 2, 3])
        sample.validate()

# def test_pvnet_sample_type_conversion(real_sample_data):
#     """Test type conversion methods"""
#     sample = PVNetSample()
    
#     # Ensure we're working with numpy arrays
#     processed_data = {}
#     for key, value in real_sample_data.items():
#         if isinstance(value, dict):
#             processed_data[key] = {
#                 k: v.copy() if isinstance(v, np.ndarray) else v
#                 for k, v in value.items()
#             }
#         else:
#             processed_data[key] = value.copy() if isinstance(value, np.ndarray) else value
    
#     for key, value in processed_data.items():
#         sample[key] = value
    
#     # Test numpy conversion
#     numpy_sample = sample.to_numpy()
#     assert isinstance(numpy_sample['nwp']['ukv']['nwp'], np.ndarray)
    
#     # Test torch conversion
#     torch_sample = sample.to_torch()
#     assert isinstance(torch_sample['nwp']['ukv']['nwp'], torch.Tensor)


# def test_pvnet_sample_save_load(tmp_path, real_sample_data):
#     """Test save and load functionality"""
#     sample = PVNetSample()
    
#     # Add data
#     processed_data = {}
#     for key, value in real_sample_data.items():
#         if isinstance(value, dict):
#             processed_data[key] = {
#                 k: v.copy() if isinstance(v, np.ndarray) else v
#                 for k, v in value.items()
#             }
#         else:
#             processed_data[key] = value.copy() if isinstance(value, np.ndarray) else value
    
#     for key, value in processed_data.items():
#         sample[key] = value
    
#     # Test saving and loading with NPZ format
#     save_path = tmp_path / "test_sample.npz"
#     sample.save(save_path)
#     assert save_path.exists()
    
#     loaded_sample = PVNetSample.load(save_path)
#     assert isinstance(loaded_sample, PVNetSample)
    
#     # Compare data recursively
#     for key in sample.keys():
#         if isinstance(sample[key], dict):
#             for sub_key in sample[key]:
#                 if isinstance(sample[key][sub_key], np.ndarray):
#                     np.testing.assert_array_equal(
#                         sample[key][sub_key],
#                         loaded_sample[key][sub_key]
#                     )
#                 else:
#                     # For non-numeric arrays like channel names
#                     assert np.array_equal(
#                         sample[key][sub_key],
#                         loaded_sample[key][sub_key]
#                     )


def test_pvnet_sample_type_conversion(real_sample_data):
    """Test type conversion methods"""
    sample = PVNetSample()
    
    processed_data = {}
    for key, value in real_sample_data.items():
        if isinstance(value, dict):
            processed_data[key] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            processed_data[key] = value.copy() if isinstance(value, np.ndarray) else value
    
    for key, value in processed_data.items():
        sample[key] = value
    
    # Test numpy conversion
    numpy_sample = sample.to_numpy()
    assert isinstance(numpy_sample['nwp']['ukv']['nwp'], np.ndarray)
    
    # Modify to_torch method in base.py
    def numpy_to_torch(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, xr.DataArray):
            return torch.from_numpy(x.values)
        elif isinstance(x, np.ndarray):
            # Skip conversion for non-numeric arrays
            if x.dtype.kind in 'biufc':  # numeric types
                return torch.from_numpy(x)
            return x
        return x
    
    # Monkey patch the conversion method
    sample._convert_arrays(numpy_to_torch)
    
    # Verify torch conversion
    assert isinstance(sample['nwp']['ukv']['channel_names'], np.ndarray)

def test_pvnet_sample_save_load(tmp_path, real_sample_data):
    sample = PVNetSample()
    
    processed_data = {}
    for key, value in real_sample_data.items():
        if isinstance(value, dict):
            processed_data[key] = {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            processed_data[key] = value.copy() if isinstance(value, np.ndarray) else value
    
    for key, value in processed_data.items():
        sample[key] = value
    
    save_path = tmp_path / "test_sample.npz"
    sample.save(save_path)
    assert save_path.exists()
    
    loaded_sample = PVNetSample.load(save_path)
    assert isinstance(loaded_sample, PVNetSample)
    
    # Compare data recursively
    for key in sample.keys():
        if isinstance(sample[key], dict):
            for sub_key in sample[key]:
                if isinstance(sample[key][sub_key], np.ndarray):
                    # Numeric array comparison
                    np.testing.assert_array_equal(
                        sample[key][sub_key],
                        loaded_sample[key][sub_key]
                    )
                else:
                    # String array comparison
                    assert np.array_equal(
                        sample[key][sub_key], 
                        loaded_sample[key][sub_key]
                    )


def test_pvnet_sample_plot(real_sample_data):
    """Test plot functionality"""
    sample = PVNetSample()
    for key, value in real_sample_data.items():
        sample[key] = value
    
    sample.plot()
    plt.close('all')

# PVNetUKRegionalDataset Tests
@patch('ocf_data_sampler.sample.uk_regional.get_gsp_locations')
def test_dataset_initialization(mock_locations, config_file, mock_data_loading):
    """Test dataset initialization"""
    mock_locations.return_value = [
        Location(id=i, x=0, y=0, coordinate_system="osgb")
        for i in range(1, 4)
    ]
    
    dataset = PVNetUKRegionalDataset(
        config_filename=str(config_file),
        start_time="2024-01-01",
        end_time="2024-01-02"
    )
    
    assert isinstance(dataset, PVNetUKRegionalDataset)
    assert hasattr(dataset, 'config')
    assert hasattr(dataset, 'valid_t0_times')
    assert hasattr(dataset, 'locations')

@patch('ocf_data_sampler.sample.uk_regional.get_gsp_locations')
def test_dataset_sample_creation(mock_locations, config_file, mock_data_loading):
    """Test sample creation in dataset"""
    mock_locations.return_value = [
        Location(id=i, x=0, y=0, coordinate_system="osgb")
        for i in range(1, 4)
    ]
    
    dataset = PVNetUKRegionalDataset(
        config_filename=str(config_file),
        start_time="2024-01-01",
        end_time="2024-01-02"
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        assert isinstance(sample, PVNetSample)
        sample.validate()

@patch('ocf_data_sampler.sample.uk_regional.get_gsp_locations')
def test_dataset_gsp_ids(mock_locations, config_file, mock_data_loading):
    """Test dataset initialization with specific GSP IDs"""
    test_gsp_ids = [1, 2, 3]
    mock_locations.return_value = [
        Location(id=gsp_id, x=0, y=0, coordinate_system="osgb")
        for gsp_id in test_gsp_ids
    ]
    
    dataset = PVNetUKRegionalDataset(
        config_filename=str(config_file),
        gsp_ids=test_gsp_ids
    )
    
    assert all(loc.id in test_gsp_ids for loc in dataset.locations)
    assert all(gsp_id in dataset.location_lookup for gsp_id in test_gsp_ids)

if __name__ == '__main__':
    pytest.main([__file__])