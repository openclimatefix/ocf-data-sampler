# test_site.py

"""
Site class testing - SiteSample
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import pandas as pd
from pathlib import Path

from ocf_data_sampler.sample.site import SiteSample


def create_test_dataset():
    """ Create synthetic xarray dataset - prefixes stated """
    times = pd.date_range('2024-01-01 00:00', periods=24, freq='h')
    data = xr.Dataset(
        data_vars={
            'site': (('site__time_utc'), np.random.rand(24)),
        },
        coords={
            'site__time_utc': times,
            'site__capacity_kwp': 100.0
        }
    )
    return data


@pytest.fixture
def sample_with_data():
    sample = SiteSample()
    sample._data = create_test_dataset()
    return sample


def test_site_sample_init():
    sample = SiteSample()
    assert isinstance(sample._data, dict)
    assert len(sample._data) == 0


def test_save_load(sample_with_data):
    """ Test save and load functionality """
    with tempfile.NamedTemporaryFile(suffix='.nc') as tf:

        # Test save
        sample_with_data.save(tf.name)
        assert Path(tf.name).exists()
        assert Path(tf.name).stat().st_size > 0
        
        # Test load
        loaded = SiteSample.load(tf.name)
        assert isinstance(loaded, SiteSample)
        assert isinstance(loaded._data, xr.Dataset)
        
        # Assert data consistency
        xr.testing.assert_identical(loaded._data, sample_with_data._data)
        np.testing.assert_array_equal(
            loaded._data['site'].values,
            sample_with_data._data['site'].values
        )

def test_to_numpy(sample_with_data):
    """ Test numpy conversion """
    numpy_data = sample_with_data.to_numpy()
    
    # Check dictionary structure
    assert isinstance(numpy_data, dict)
    assert 'site' in numpy_data
    
    # Validate array
    assert isinstance(numpy_data['site'], np.ndarray)
    assert numpy_data['site'].shape == (24,)
    assert not np.any(np.isnan(numpy_data['site']))
    
    # Verify data range
    assert np.all((numpy_data['site'] >= 0) & (numpy_data['site'] <= 1))


def test_invalid_save_format(sample_with_data):
    with tempfile.NamedTemporaryFile(suffix='.txt') as tf:
        with pytest.raises(ValueError, match="Only .nc format is supported"):
            sample_with_data.save(tf.name)


def test_invalid_load_format():
    with tempfile.NamedTemporaryFile(suffix='.txt') as tf:
        with pytest.raises(ValueError, match="Only .nc format is supported"):
            SiteSample.load(tf.name)


def test_invalid_data_type():
    sample = SiteSample()
    sample._data = {"invalid": "data"}
    
    with pytest.raises(TypeError, match="Data must be xarray Dataset"):
        sample.to_numpy()
        
    with pytest.raises(TypeError, match="Data must be xarray Dataset for saving"):
        with tempfile.NamedTemporaryFile(suffix='.nc') as tf:
            sample.save(tf.name)


def test_corrupted_file():
    with tempfile.NamedTemporaryFile(suffix='.nc') as tf:
        with open(tf.name, 'wb') as f:
            f.write(b'corrupted data')
        with pytest.raises(Exception):
            SiteSample.load(tf.name)
