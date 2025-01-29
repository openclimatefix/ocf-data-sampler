"""
UK Regional class testing - UKRegionalSample
"""

import pytest
import numpy as np
import torch
import tempfile

from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    SatelliteSampleKey,
    NWPSampleKey
)

from ocf_data_sampler.sample.uk_regional import UKRegionalSample


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


# UKRegionalSample testing
def test_sample_init():
    """ Initialisation """
    sample = UKRegionalSample()
    assert hasattr(sample, '_data'), "Sample should have _data attribute"
    assert isinstance(sample._data, dict)
    assert len(sample._data) == 0


def test_sample_save_load():
   sample = UKRegionalSample()
   sample._data = create_test_data()
   
   with tempfile.NamedTemporaryFile(suffix='.pt') as tf:
       sample.save(tf.name)
       loaded = UKRegionalSample.load(tf.name)
       
       assert set(loaded._data.keys()) == set(sample._data.keys())
       assert isinstance(loaded._data['nwp'], dict)
       assert 'ukv' in loaded._data['nwp']

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
   sample = UKRegionalSample()
   sample._data = create_test_data()
  
   with tempfile.NamedTemporaryFile(suffix='.npz') as tf:
       with pytest.raises(ValueError, match="Only .pt format is supported"):
           sample.save(tf.name)


def test_load_unsupported_format():
    """ Test loading - unsupported file format """

    with tempfile.NamedTemporaryFile(suffix='.npz') as tf:
        with pytest.raises(ValueError, match="Only .pt format is supported"):
            UKRegionalSample.load(tf.name)


def test_load_corrupted_file():
    """ Test loading - corrupted / empty file """

    with tempfile.NamedTemporaryFile(suffix='.pt') as tf:
        with open(tf.name, 'wb') as f:
            f.write(b'corrupted data')
        
        with pytest.raises(Exception):
            UKRegionalSample.load(tf.name)


def test_to_numpy():
    """ To numpy conversion check """
    sample = UKRegionalSample()
    sample._data = {
        'nwp': {
            'ukv': {
                'nwp': np.random.rand(4, 1, 2, 2),
                'x': np.array([1, 2]),
                'y': np.array([1, 2])
            }
        },
        GSPSampleKey.gsp: np.random.rand(7),
        SatelliteSampleKey.satellite_actual: np.random.rand(7, 1, 2, 2),
        GSPSampleKey.solar_azimuth: np.random.rand(7),
        GSPSampleKey.solar_elevation: np.random.rand(7)
    }
    
    numpy_data = sample.to_numpy()
    
    # Check returned data matches
    assert numpy_data == sample._data
    assert len(numpy_data) == len(sample._data)
    
    # Assert specific keys and types
    assert 'nwp' in numpy_data
    assert isinstance(numpy_data['nwp']['ukv']['nwp'], np.ndarray)
    assert numpy_data[GSPSampleKey.gsp].shape == (7,)
