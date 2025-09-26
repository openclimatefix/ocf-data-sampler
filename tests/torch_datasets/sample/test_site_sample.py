"""
Site class testing - SiteSample
"""

import tempfile

import numpy as np

from ocf_data_sampler.numpy_sample import SatelliteSampleKey, SiteSampleKey
from ocf_data_sampler.torch_datasets.sample.site import SiteSample


def test_site_sample_with_data(numpy_sample_site):
    """Testing of defined sample with actual data"""
    sample = SiteSample(numpy_sample_site)

    assert isinstance(sample._data, dict)
    assert sample._data["satellite_actual"].shape == (7, 1, 2, 2)
    assert sample._data["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    assert sample._data["site"].shape == (7,)
    assert sample._data["solar_azimuth"].shape == (7,)
    assert sample._data["date_sin"].shape == (7,)


def test_sample_save_load(numpy_sample_site):
    sample = SiteSample(numpy_sample_site)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tf:
        sample.save(tf.name)
        loaded = SiteSample.load(tf.name)

        assert set(loaded._data) == set(sample._data)
        assert isinstance(loaded._data["nwp"], dict)
        assert "ukv" in loaded._data["nwp"]
        assert loaded._data[SiteSampleKey.generation].shape == (7,)
        assert loaded._data[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)

        np.testing.assert_array_almost_equal(
            loaded._data[SiteSampleKey.generation],
            sample._data[SiteSampleKey.generation],
        )


def test_to_numpy(numpy_sample_site):
    """To numpy conversion"""
    sample = SiteSample(numpy_sample_site)
    numpy_data = sample.to_numpy()

    assert isinstance(numpy_data, dict)
    assert "site" in numpy_data and "nwp" in numpy_data

    # Check site - numpy array instead of dict
    site_data = numpy_data["site"]
    assert isinstance(site_data, np.ndarray)
    assert site_data.ndim == 1
    assert len(site_data) == 7
    assert np.all((site_data >= 0) & (site_data <= 1))

    # Check NWP
    assert "ukv" in numpy_data["nwp"]
    nwp_data = numpy_data["nwp"]["ukv"]
    assert "nwp" in nwp_data
    assert nwp_data["nwp"].shape == (4, 1, 2, 2)
