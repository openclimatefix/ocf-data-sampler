"""
Site class testing - SiteSample
"""
import tempfile

import numpy as np
import pytest

from ocf_data_sampler.numpy_sample import NWPSampleKey, SatelliteSampleKey, SiteSampleKey
from ocf_data_sampler.torch_datasets.sample.site import SiteSample


@pytest.fixture
def numpy_sample():
    """Synthetic data generation"""
    expected_site_shape = (7,)
    expected_nwp_ukv_shape = (4, 1, 2, 2)
    expected_sat_shape = (7, 1, 2, 2)
    expected_solar_shape = (7,)

    nwp_data = {
        "nwp": np.random.rand(*expected_nwp_ukv_shape),
        "x": np.array([1, 2]),
        "y": np.array([1, 2]),
        NWPSampleKey.channel_names: ["t"],
    }

    return {
        "nwp": {
            "ukv": nwp_data,
        },
        SiteSampleKey.generation: np.random.rand(*expected_site_shape),
        SatelliteSampleKey.satellite_actual: np.random.rand(*expected_sat_shape),
        "solar_azimuth": np.random.rand(*expected_solar_shape),
        "solar_elevation": np.random.rand(*expected_solar_shape),
        "date_cos": np.random.rand(*expected_solar_shape),
        "date_sin": np.random.rand(*expected_solar_shape),
        "time_cos": np.random.rand(*expected_solar_shape),
        "time_sin": np.random.rand(*expected_solar_shape),
    }


def test_site_sample_with_data(numpy_sample):
    """Testing of defined sample with actual data"""
    sample = SiteSample(numpy_sample)

    # Assert data structure
    assert isinstance(sample._data, dict)

    assert sample._data["satellite_actual"].shape == (7, 1, 2, 2)
    assert sample._data["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    assert sample._data["site"].shape == (7,)
    assert sample._data["solar_azimuth"].shape == (7,)
    assert sample._data["date_sin"].shape == (7,)


def test_sample_save_load(numpy_sample):
    sample = SiteSample(numpy_sample)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tf:
        sample.save(tf.name)
        loaded = SiteSample.load(tf.name)

        assert set(loaded._data.keys()) == set(sample._data.keys())
        assert isinstance(loaded._data["nwp"], dict)
        assert "ukv" in loaded._data["nwp"]

        assert loaded._data[SiteSampleKey.generation].shape == (7,)
        assert loaded._data[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)

        np.testing.assert_array_almost_equal(
            loaded._data[SiteSampleKey.generation],
            sample._data[SiteSampleKey.generation],
        )


def test_to_numpy(numpy_sample):
    """To numpy conversion"""
    sample = SiteSample(numpy_sample)
    numpy_data = sample.to_numpy()

    # Assert structure
    assert isinstance(numpy_data, dict)
    assert "site" in numpy_data
    assert "nwp" in numpy_data

    # Check site - numpy array instead of dict
    site_data = numpy_data["site"]
    assert isinstance(site_data, np.ndarray)
    assert site_data.ndim == 1
    assert len(site_data) == 7
    assert site_data.dtype == np.float32
    assert np.all(site_data >= 0) 

    # Check NWP
    assert "ukv" in numpy_data["nwp"]
    nwp_data = numpy_data["nwp"]["ukv"]
    assert "nwp" in nwp_data
    assert nwp_data["nwp"].shape == (4, 1, 2, 2)
