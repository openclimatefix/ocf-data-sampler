# test_uk_regional.py

"""
UK Regional class testing - PVNetSample / PVNetUKRegionalDataset
"""

import pytest
import numpy as np
import torch
import tempfile

from pathlib import Path

from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    SatelliteSampleKey,
    NWPSampleKey
)

from ocf_data_sampler.torch_datasets.pvnet_uk_regional import PVNetUKRegionalDataset

from ocf_data_sampler.sample.uk_regional import PVNetSample


# Fixture define
@pytest.fixture
def pvnet_config_filename(tmp_path):
    """ Minimal config file - testing """
    config_content = """
    input_data:
        gsp:
            zarr_path: ""
            time_resolution_minutes: 30
            interval_start_minutes: -180
            interval_end_minutes: 0
        nwp:
            ukv:
                zarr_path: ""
                image_size_pixels_height: 64
                image_size_pixels_width: 64
                time_resolution_minutes: 60
                interval_start_minutes: -180
                interval_end_minutes: 0
                channels: ["t", "dswrf"]
                provider: "ukv"
        satellite:
            zarr_path: ""
            image_size_pixels_height: 64
            image_size_pixels_width: 64
            time_resolution_minutes: 30
            interval_start_minutes: -180
            interval_end_minutes: 0
            channels: ["HRV"]
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


def create_test_data():
    """ Synthetic data generation """

    # Field / spatial coordinates
    nwp_data = {
        'nwp': np.random.rand(4, 1, 2, 2),
        'x': np.array([1, 2]),
        'y': np.array([1, 2]),
        NWPSampleKey.channel_names: ['test_channel']
    }
    
    return {
        'nwp': {
            'ukv': nwp_data
        },
        GSPSampleKey.gsp: np.random.rand(7),
        SatelliteSampleKey.satellite_actual: np.random.rand(7, 1, 2, 2),
        GSPSampleKey.solar_azimuth: np.random.rand(7),
        GSPSampleKey.solar_elevation: np.random.rand(7)
    }


# PVNetSample testing
def test_pvnet_sample_init():
    """ Initialisation / validation """

    sample = PVNetSample()
    
    # Validate empty feature space
    with pytest.raises(ValueError):
        sample.validate()
    
    # Test non empty feature space
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
    
    # Validate non empty feature space
    sample.validate()


def test_pvnet_sample_invalid_nwp():
    """ Test invalid NWP structure validation """

    sample = PVNetSample()
    test_data = create_test_data()
    
    test_data['nwp'] = np.random.rand(4, 1, 2, 2)
    
    for key, value in test_data.items():
        sample[key] = value
        
    with pytest.raises(TypeError, match="NWP data must be nested dictionary"):
        sample.validate()


def test_pvnet_sample_save_load():
    """ Save / load functionality """

    sample = PVNetSample()
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
    
    # Persistence in .pt format
    with tempfile.NamedTemporaryFile(suffix='.pt') as tf:
        sample.save(tf.name)
        loaded = PVNetSample.load(tf.name)
        
        assert set(loaded.keys()) == set(sample.keys())
        
        # Verify NWP structure
        assert isinstance(loaded['nwp'], dict)
        assert 'ukv' in loaded['nwp']

        # Verify other key shapes / consistency
        assert loaded[GSPSampleKey.gsp].shape == (7,)
        assert loaded[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
        assert loaded[GSPSampleKey.solar_azimuth].shape == (7,)
        assert loaded[GSPSampleKey.solar_elevation].shape == (7,)

        np.testing.assert_array_almost_equal(
            loaded[GSPSampleKey.gsp],
            sample[GSPSampleKey.gsp]
        )


def test_save_unsupported_format():
    """ Test saving - unsupported file format """

    sample = PVNetSample()
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
    
    with tempfile.NamedTemporaryFile(suffix='.npz') as tf:
        with pytest.raises(ValueError, match="Only .pt format is supported"):
            sample.save(tf.name)


def test_load_unsupported_format():
    """ Test loading - unsupported file format """

    with tempfile.NamedTemporaryFile(suffix='.npz') as tf:
        with pytest.raises(ValueError, match="Only .pt format is supported"):
            PVNetSample.load(tf.name)


def test_load_corrupted_file():
    """ Test loading - corrupted / empty file """

    with tempfile.NamedTemporaryFile(suffix='.pt') as tf:
        with open(tf.name, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(Exception):
            PVNetSample.load(tf.name)


def test_pvnet_sample_invalid_shapes():
    """ Test validation of inconsistent array shapes """

    sample = PVNetSample()
    test_data = create_test_data()
    
    # Modify one array to have inconsistent time steps
    test_data[GSPSampleKey.gsp] = np.random.rand(6)
    
    for key, value in test_data.items():
        sample[key] = value
    with pytest.raises(ValueError, match="Inconsistent number of timesteps"):
        sample.validate()


# PVNetUKRegionalDataset testing
def test_pvnet_dataset(pvnet_config_filename):
    """ Dataset initialisation """

    # Temporal domain defined
    start_time = "2024-01-01 00:00:00"
    end_time = "2024-01-02 00:00:00"
    
    with pytest.raises(Exception):
        dataset = PVNetUKRegionalDataset(
            pvnet_config_filename,
            start_time=start_time,
            end_time=end_time
        )


def test_pvnet_dataset_get_sample(pvnet_config_filename):
    """ Direct sample access - time and GSP ID """

    # Empty spatial domain
    with pytest.raises(Exception):
        dataset = PVNetUKRegionalDataset(pvnet_config_filename)


def test_pvnet_sample_to_numpy():
    """ Test conversion of sample data to numpy arrays """
    
    sample = PVNetSample()
    
    # Create mixed data types for testing
    mixed_data = {
        'nwp': {
            'ukv': {
                'nwp': torch.rand(4, 1, 2, 2),
                'x': np.array([1, 2]),
                'y': np.array([1, 2])
            }
        },
        GSPSampleKey.gsp: torch.rand(7),
        SatelliteSampleKey.satellite_actual: torch.rand(7, 1, 2, 2),
        GSPSampleKey.solar_azimuth: np.random.rand(7),
        GSPSampleKey.solar_elevation: np.random.rand(7)
    }
    
    for key, value in mixed_data.items():
        sample[key] = value
    
    numpy_data = sample.to_numpy()
    
    # Verify all data is numpy arrays
    assert isinstance(numpy_data[GSPSampleKey.gsp], np.ndarray)
    assert isinstance(numpy_data[SatelliteSampleKey.satellite_actual], np.ndarray)
    assert isinstance(numpy_data[GSPSampleKey.solar_azimuth], np.ndarray)
    assert isinstance(numpy_data[GSPSampleKey.solar_elevation], np.ndarray)
    
    # Verify nested NWP structure
    assert isinstance(numpy_data['nwp']['ukv']['nwp'], np.ndarray)
    assert isinstance(numpy_data['nwp']['ukv']['x'], np.ndarray)
    assert isinstance(numpy_data['nwp']['ukv']['y'], np.ndarray)
    
    # Verify shapes are preserved
    assert numpy_data[GSPSampleKey.gsp].shape == (7,)
    assert numpy_data[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
    assert numpy_data['nwp']['ukv']['nwp'].shape == (4, 1, 2, 2)
