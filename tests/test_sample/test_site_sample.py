# test_site_sample.py

"""
Site class testing - SiteSample
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import pandas as pd

from pathlib import Path

from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.sample.site import SiteSample


@pytest.fixture
def sample_with_data():
    """ Sample - matches expected structure """

    sample = SiteSample()
    
    # Base time periods specified
    init_time = pd.Timestamp("2023-01-01 00:00")
    target_times = pd.date_range("2023-01-01 00:00", periods=4, freq="1h")
    sat_times = pd.date_range("2023-01-01 00:00", periods=7, freq="5min")
    site_times = pd.date_range("2023-01-01 00:00", periods=4, freq="30min")

    # Specify steps for NWP data
    steps = [(t - init_time) for t in target_times]
    
    data = xr.Dataset(
        data_vars={
            'nwp-ukv': (
                [
                    "nwp-ukv__target_time_utc",
                    "nwp-ukv__channel",
                    "nwp-ukv__y_osgb",
                    "nwp-ukv__x_osgb"
                ],
                np.random.rand(4, 1, 2, 2)
            ),
            'satellite': (
                [
                    "satellite__time_utc",
                    "satellite__channel",
                    "satellite__y_geostationary",
                    "satellite__x_geostationary"
                ],
                np.random.rand(7, 1, 2, 2)
            ),
            'site': (["site__time_utc"], np.random.rand(4))
        },
        coords={
            # NWP coords
            'nwp-ukv__target_time_utc': target_times,
            'nwp-ukv__channel': ['dswrf'],
            'nwp-ukv__y_osgb': [0, 1],
            'nwp-ukv__x_osgb': [0, 1],
            'nwp-ukv__init_time_utc': init_time,
            'nwp-ukv__step': ('nwp-ukv__target_time_utc', steps),  # Add step coordinate
            
            # Sat coords
            'satellite__time_utc': sat_times,
            'satellite__channel': ['vis'],
            'satellite__y_geostationary': [0, 1],
            'satellite__x_geostationary': [0, 1],
            
            # Site specific coords
            'site__time_utc': site_times,
            'site__capacity_kwp': 1000.0,
            'site__site_id': 1,
            'site__longitude': -3.5,
            'site__latitude': 51.5,
            
            # Site features as coords
            'site__solar_azimuth': ('site__time_utc', np.random.rand(4)),
            'site__solar_elevation': ('site__time_utc', np.random.rand(4)),
            'site__date_cos': ('site__time_utc', np.random.rand(4)),
            'site__date_sin': ('site__time_utc', np.random.rand(4)),
            'site__time_cos': ('site__time_utc', np.random.rand(4)),
            'site__time_sin': ('site__time_utc', np.random.rand(4))
        }
    )
    
    sample._data = data
    return sample


def test_site_sample_init():
    """ Basic initialisation """
    sample = SiteSample()
    assert isinstance(sample._data, dict)
    assert len(sample._data) == 0


def test_save_load(sample_with_data):
    """ Standard save and load functionality """
    with tempfile.NamedTemporaryFile(suffix='.nc') as tf:

        sample_with_data.save(tf.name)
        assert Path(tf.name).exists()
        assert Path(tf.name).stat().st_size > 0        
        loaded = SiteSample.load(tf.name)
        assert isinstance(loaded, SiteSample)
        assert isinstance(loaded._data, xr.Dataset)
        
        # Assert data structure - in line with torch_dataset testing
        expected_dims = {
            "satellite__x_geostationary",
            "site__time_utc",
            "nwp-ukv__target_time_utc",
            "nwp-ukv__x_osgb",
            "satellite__channel",
            "satellite__y_geostationary",
            "satellite__time_utc",
            "nwp-ukv__channel",
            "nwp-ukv__y_osgb",
        }
        assert set(loaded._data.dims) == expected_dims

        # Assert shapes match
        assert loaded._data["satellite"].values.shape == (7, 1, 2, 2)
        assert loaded._data["nwp-ukv"].values.shape == (4, 1, 2, 2)
        assert loaded._data["site"].values.shape == (4,)


def test_invalid_save_format(sample_with_data):
    """ Saving with invalid format """
    with tempfile.NamedTemporaryFile(suffix='.txt') as tf:
        with pytest.raises(ValueError, match="Only .nc format is supported"):
            sample_with_data.save(tf.name)


def test_invalid_load_format():
    """ Loading with invalid format """
    with tempfile.NamedTemporaryFile(suffix='.txt') as tf:
        with pytest.raises(ValueError, match="Only .nc format is supported"):
            SiteSample.load(tf.name)


def test_invalid_data_type():
    """ Invalid data type """
    sample = SiteSample()
    sample._data = {"invalid": "data"}
    
    with pytest.raises(TypeError, match="Data must be xarray Dataset"):
        sample.to_numpy()
        
    with pytest.raises(TypeError, match="Data must be xarray Dataset for saving"):
        with tempfile.NamedTemporaryFile(suffix='.nc') as tf:
            sample.save(tf.name)


def test_corrupted_file():
    """ Loading corrupted file """
    with tempfile.NamedTemporaryFile(suffix='.nc') as tf:
        with open(tf.name, 'wb') as f:
            f.write(b'corrupted data')
        with pytest.raises(Exception):
            SiteSample.load(tf.name)


def test_to_numpy(sample_with_data):
    """ To numpy conversion """
    numpy_data = sample_with_data.to_numpy()    
    
    # Basic structure assertions
    assert isinstance(numpy_data, dict), "Numpy data must be dict"
    assert 'site' in numpy_data, "Numpy data must contain 'site' key"
    assert 'nwp' in numpy_data, "Numpy data must contain 'nwp' key"
    assert isinstance(numpy_data['site'], dict), "Site data must be dict"
    
    # Check for required keys in site data
    site_data = numpy_data['site']    
    assert len(site_data) > 0, "Site data dict must not be empty"
    
    # Verify site data contents
    for key, value in site_data.items():
        assert isinstance(value, np.ndarray), f"Site data value for {key} must be NumPy array"
        assert value.ndim == 1, f"Site data value for {key} must be 1-D array"
        assert value.shape[0] == 4, f"Site data value for {key} must have 4 elements"
        assert np.issubdtype(value.dtype, np.floating), f"Site data value for {key} must contain floating point values"
        assert np.all(value >= 0), f"Site data values for {key} must be non negative"
        assert np.all(value <= 1), f"Site data values for {key} must be less than or equal to 1"

    # Additional NWP data assertions
    assert 'ukv' in numpy_data['nwp'], "NWP data must contain 'ukv' key"
    nwp_data = numpy_data['nwp']['ukv']
    assert 'nwp' in nwp_data, "NWP data must contain 'nwp' key"
    assert isinstance(nwp_data['nwp'], np.ndarray), "NWP value must be a NumPy array"
    assert nwp_data['nwp'].shape == (4, 1, 2, 2), "NWP array must have shape (4, 1, 2, 2)"
