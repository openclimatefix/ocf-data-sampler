# test_uk_regional.py

"""Tests for PVNet sample and dataset implementations using real data"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from ocf_data_sampler.sample.uk_regional import PVNetSample, PVNetUKRegionalDataset
from ocf_data_sampler.numpy_batch import (
    NWPBatchKey,
    GSPBatchKey,
    SatelliteBatchKey
)
from ocf_data_sampler.select import Location
from ocf_data_sampler.config import Configuration


@pytest.fixture
def real_sample_data():
    """Create realistic sample data for testing"""
    # Create NWP-like data
    nwp_data = xr.DataArray(
        np.random.rand(24, 64, 64),
        dims=['time', 'y', 'x'],
        coords={
            'time': pd.date_range('2024-01-01', periods=24, freq='1H'),
            'y': np.arange(64),
            'x': np.arange(64)
        }
    )
    
    # Create GSP-like data with realistic dimensions and timing
    gsp_data = np.random.rand(48)  # 48 half-hourly values
    
    # Create satellite-like data
    sat_data = np.random.rand(64, 64)  # Realistic satellite image dimensions
    
    # Create solar position data
    solar_data = np.random.rand(48)  # Matching GSP timeline
    
    return {
        'nwp': {
            'ukv': {
                'nwp': nwp_data,
                'channel_names': ['temperature', 'precipitation', 'radiation']
            }
        },
        GSPBatchKey.gsp: gsp_data,
        SatelliteBatchKey.satellite_actual: sat_data,
        GSPBatchKey.solar_azimuth: solar_data,
        GSPBatchKey.solar_elevation: solar_data
    }

@pytest.fixture
def sample_config():
    """Create a real configuration for testing"""
    config_dict = {
        'input_data': {
            'gsp': {
                'time_resolution_minutes': 30,
                'interval_start_minutes': 0,
                'interval_end_minutes': 1440
            },
            'nwp': {
                'ukv': {
                    'time_resolution_minutes': 60,
                    'interval_start_minutes': 0,
                    'interval_end_minutes': 1440,
                    'provider': 'ukv'
                }
            },
            'satellite': {
                'time_resolution_minutes': 15,
                'interval_start_minutes': 0,
                'interval_end_minutes': 1440
            }
        }
    }
    return Configuration.from_dict(config_dict)

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
    incomplete_data = real_sample_data.copy()
    del incomplete_data[GSPBatchKey.gsp]
    for key, value in incomplete_data.items():
        sample[key] = value
    with pytest.raises(ValueError, match="Missing required keys"):
        sample.validate()

def test_pvnet_sample_type_conversion(real_sample_data):
    """Test type conversion methods"""
    sample = PVNetSample()
    for key, value in real_sample_data.items():
        sample[key] = value
    
    # Test numpy conversion
    numpy_sample = sample.to_numpy()
    assert isinstance(numpy_sample['nwp']['ukv']['nwp'], np.ndarray)
    
    # Test torch conversion
    torch_sample = sample.to_torch()
    import torch
    assert isinstance(torch_sample['nwp']['ukv']['nwp'], torch.Tensor)

def test_pvnet_sample_save_load(tmp_path, real_sample_data):
    """Test save and load functionality"""
    sample = PVNetSample()
    for key, value in real_sample_data.items():
        sample[key] = value
    
    # Test saving and loading with different formats
    for format_suffix in ['.npz', '.nc', '.zarr']:
        save_path = tmp_path / f"test_sample{format_suffix}"
        
        # Save
        sample.save(save_path)
        assert save_path.exists()
        
        # Load
        loaded_sample = PVNetSample.load(save_path)
        assert isinstance(loaded_sample, PVNetSample)
        
        # Compare data
        for key in sample.keys():
            if isinstance(sample[key], dict):
                for sub_key in sample[key]:
                    np.testing.assert_array_equal(
                        sample[key][sub_key],
                        loaded_sample[key][sub_key]
                    )
            else:
                np.testing.assert_array_equal(
                    sample[key],
                    loaded_sample[key]
                )

# PVNetUKRegionalDataset Tests
def test_dataset_initialization(tmp_path, sample_config):
    """Test dataset initialization with real config"""
    # Create a temporary config file
    config_path = tmp_path / "test_config.yaml"
    sample_config.save(config_path)
    
    # Initialize dataset
    dataset = PVNetUKRegionalDataset(
        config_filename=str(config_path),
        start_time="2024-01-01",
        end_time="2024-01-02"
    )
    
    assert isinstance(dataset, PVNetUKRegionalDataset)
    assert hasattr(dataset, 'config')
    assert hasattr(dataset, 'valid_t0_times')
    assert hasattr(dataset, 'locations')

def test_dataset_sample_creation(tmp_path, sample_config):
    """Test sample creation in dataset"""
    # Create config file
    config_path = tmp_path / "test_config.yaml"
    sample_config.save(config_path)
    
    dataset = PVNetUKRegionalDataset(
        config_filename=str(config_path),
        start_time="2024-01-01",
        end_time="2024-01-02"
    )
    
    # Test getting a sample
    if len(dataset) > 0:
        sample = dataset[0]
        assert isinstance(sample, PVNetSample)
        
        # Test sample validity
        sample.validate()

if __name__ == '__main__':
    pytest.main([__file__])
    