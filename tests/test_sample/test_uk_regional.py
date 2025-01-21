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
    
    # Test initialisation with data and validate
    test_data = create_test_data()
    sample = PVNetSample(data=test_data)
    sample.validate()


def test_pvnet_sample_invalid_nwp():
    """ Test invalid NWP structure validation """
    test_data = create_test_data()
    test_data['nwp'] = np.random.rand(4, 1, 2, 2)
    
    sample = PVNetSample(data=test_data)
    with pytest.raises(TypeError, match="NWP data must be nested dictionary"):
        sample.validate()


def test_pvnet_sample_save_load():
    """ Save / load functionality """
    test_data = create_test_data()
    sample = PVNetSample(data=test_data)
    
    # Persistence in .pt format
    with tempfile.NamedTemporaryFile(suffix='.pt') as tf:
        sample.save(tf.name)
        loaded = PVNetSample.load(tf.name)
        
        assert set(loaded.keys()) == set(sample.keys())
        
        # Verify NWP structure
        assert isinstance(loaded._data['nwp'], dict)
        assert 'ukv' in loaded._data['nwp']

        # Verify other key shapes / consistency
        assert loaded._data[GSPSampleKey.gsp].shape == (7,)
        assert loaded._data[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
        assert loaded._data[GSPSampleKey.solar_azimuth].shape == (7,)
        assert loaded._data[GSPSampleKey.solar_elevation].shape == (7,)

        np.testing.assert_array_almost_equal(
            loaded._data[GSPSampleKey.gsp],
            sample._data[GSPSampleKey.gsp]
        )


def test_save_unsupported_format():
   """ Test saving - unsupported file format """
   test_data = create_test_data()
   sample = PVNetSample(data=test_data)
   
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
   
    # Modify one to have inconsistent time steps
    test_data = create_test_data()
    test_data[GSPSampleKey.gsp] = np.random.rand(6)
   
    sample = PVNetSample(data=test_data)
   
    with pytest.raises(ValueError, match="Inconsistent number of timesteps"):
        sample.validate()


# PVNetUKRegionalDataset testing
def test_pvnet_dataset_get_sample(pvnet_config_filename):
    """ Direct sample access - time and GSP ID """

    # Empty spatial domain
    with pytest.raises(Exception):
        dataset = PVNetUKRegionalDataset(pvnet_config_filename)


def test_pvnet_sample_to_numpy():
   """ Test conversion of sample data to numpy arrays """
   
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
   
   # Initialize sample with mixed data
   sample = PVNetSample(data=mixed_data)
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


# NaN related testing and further validation check
def test_pvnet_sample_nan_in_nwp():
   """ Test validation catches NaN in NWP data """
   test_data = create_test_data()
   
   # Insert NaN into NWP data
   test_data['nwp']['ukv']['nwp'][0, 0, 0, 0] = np.nan
   
   sample = PVNetSample(data=test_data)
   with pytest.raises(ValueError, match="NaN values in NWP data for ukv"):
       sample.validate()


def test_pvnet_sample_nan_in_gsp():
   """ Test validation catches NaN in GSP data """
   test_data = create_test_data()
   
   # Insert NaN into GSP data 
   test_data[GSPSampleKey.gsp][0] = np.nan
   
   sample = PVNetSample(data=test_data)
   with pytest.raises(ValueError, match="NaN values in GSP data"):
       sample.validate()


def test_pvnet_sample_nan_in_time():
   """ Test validation catches NaN in time dependent data """
   test_data = create_test_data()
   
   # Insert NaN into solar azimuth data
   test_data[GSPSampleKey.solar_azimuth][0] = np.nan
   
   sample = PVNetSample(data=test_data)
   with pytest.raises(ValueError, match=f"NaN values in {GSPSampleKey.solar_azimuth}"):
       sample.validate()


def test_pvnet_sample_without_gsp():
    """ Test validation works without GSP data """
    test_data = create_test_data()
    
    # Remove optional data in a clearer way
    optional_keys = [
        GSPSampleKey.gsp,
        GSPSampleKey.solar_azimuth, 
        GSPSampleKey.solar_elevation,
        SatelliteSampleKey.satellite_actual
    ]
    
    for key in optional_keys:
        if key in test_data:
            del test_data[key]
    
    # Initialize with only required data (NWP)
    sample = PVNetSample(data=test_data)
    sample.validate()
