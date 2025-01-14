# test_uk_regional.py

"""
UK Regional class testing - PVNetSample / PVNetUKRegionalDataset
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path

from ocf_data_sampler.numpy_sample import (
    NWPSampleKey,
    GSPSampleKey,
    SatelliteSampleKey
)
from ocf_data_sampler.select import Location
from ocf_data_sampler.sample.uk_regional import PVNetSample, PVNetUKRegionalDataset


# Config fixture definition
@pytest.fixture
def pvnet_config_filename(tmp_path):
    """ Minimal config file for testing """

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
        satellite:
            zarr_path: ""
    """
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


# Modified test data creation
def create_test_data():
    """ Synthetic data generation """

    # Field and spatial coordinates
    nwp_data = {
        'nwp': np.random.rand(4, 1, 2, 2),
        'x': np.array([1, 2]),
        'y': np.array([1, 2])
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
    """ Initialisation and validation """
    sample = PVNetSample()
    
    # Validate empty feature space
    with pytest.raises(ValueError):
        sample.validate()
    
    # Test non-empty feature space
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
    
    # Validate non-empty feature space
    sample.validate()


def test_pvnet_sample_type_conversion():
    """ Feature space type conversion """
    sample = PVNetSample()
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
        
    # Test numpy conversion
    numpy_sample = sample.to_numpy()
    assert isinstance(numpy_sample['nwp']['ukv']['nwp'], np.ndarray)
    
    # Test torch conversion
    torch_sample = sample.to_torch()
    assert isinstance(torch_sample['nwp']['ukv']['nwp'], torch.Tensor)


def test_pvnet_sample_save_load():
    """ Save / load functionality """
    sample = PVNetSample()
    test_data = create_test_data()
    for key, value in test_data.items():
        sample[key] = value
    
    # Persistence in npz format
    with tempfile.NamedTemporaryFile(suffix='.npz') as tf:
        sample.save(tf.name)
        loaded = PVNetSample.load(tf.name)
        
       # Validate feature space topology
        assert set(loaded.keys()) == set(sample.keys())
        
        # Verify NWP structure
        assert isinstance(loaded['nwp'], dict)
        assert 'ukv' in loaded['nwp']

        # Verify other key shapes (dimensional consistency)
        assert loaded[GSPSampleKey.gsp].shape == (7,)
        assert loaded[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
        assert loaded[GSPSampleKey.solar_azimuth].shape == (7,)
        assert loaded[GSPSampleKey.solar_elevation].shape == (7,)

        # Test content equality for simple array first
        # Numerical equivalence validation
        np.testing.assert_array_almost_equal(
            loaded[GSPSampleKey.gsp],
            sample[GSPSampleKey.gsp]
        )


# PVNetUKRegionalDataset testing
def test_pvnet_dataset(pvnet_config_filename):
    """ Dataset initialisation """

    # Create dataset with temporal domain defined
    start_time = "2024-01-01 00:00:00"
    end_time = "2024-01-02 00:00:00"
    
    # This might raise an error pending updates 
    # Empty config currently considered
    # Potentially further mock certain data loading functions ???
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
