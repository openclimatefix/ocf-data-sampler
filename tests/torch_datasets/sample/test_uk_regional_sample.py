"""
UK Regional class testing - UKRegionalSample
"""

import tempfile

import numpy as np
import pytest

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.numpy_sample import GSPSampleKey, NWPSampleKey, SatelliteSampleKey
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample


@pytest.fixture
def numpy_sample():
    """Synthetic data generation"""
    expected_gsp_shape = (7,)
    expected_nwp_ukv_shape = (4, 1, 2, 2)
    expected_sat_shape = (7, 1, 2, 2)

    # Create NWP data with the expected shape
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
        GSPSampleKey.gsp: np.random.rand(*expected_gsp_shape),
        SatelliteSampleKey.satellite_actual: np.random.rand(*expected_sat_shape),
    }


@pytest.fixture
def pvnet_configuration_object(pvnet_config_filename) -> Configuration:
    """Loads the configuration from the temporary file path."""
    return load_yaml_configuration(pvnet_config_filename)


def test_sample_save_load(numpy_sample):
    sample = UKRegionalSample(numpy_sample)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tf:
        sample.save(tf.name)
        loaded = UKRegionalSample.load(tf.name)

        assert set(loaded._data.keys()) == set(sample._data.keys())
        assert isinstance(loaded._data["nwp"], dict)
        assert "ukv" in loaded._data["nwp"]

        assert loaded._data[GSPSampleKey.gsp].shape == (7,)
        assert loaded._data[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)

        np.testing.assert_array_almost_equal(
            loaded._data[GSPSampleKey.gsp],
            sample._data[GSPSampleKey.gsp],
        )


def test_load_corrupted_file():
    """Test loading - corrupted / empty file"""

    with tempfile.NamedTemporaryFile(suffix=".pt") as tf, open(tf.name, "wb") as f:
        f.write(b"corrupted data")

        with pytest.raises(EOFError):
            UKRegionalSample.load(tf.name)


def test_to_numpy(numpy_sample):
    """To numpy conversion check"""

    sample = UKRegionalSample(numpy_sample)

    numpy_data = sample.to_numpy()

    # Check returned data matches
    assert numpy_data == sample._data
    assert len(numpy_data) == len(sample._data)

    # Assert specific keys and types
    assert "nwp" in numpy_data
    assert isinstance(numpy_data["nwp"]["ukv"]["nwp"], np.ndarray)
    assert numpy_data[GSPSampleKey.gsp].shape == (7,)


def test_validate_sample(numpy_sample, pvnet_configuration_object: Configuration):
    """Test the validate_sample method succeeds with a loaded Configuration object."""
    sample = UKRegionalSample(numpy_sample)
    result = sample.validate_sample(pvnet_configuration_object)    
    
    # Assert success with config object
    assert isinstance(result, dict)
    assert result["status"] == "success"


def test_validate_sample_with_missing_keys(numpy_sample, pvnet_configuration_object: Configuration):
    """Test validation raises ValueError when configured satellite data is missing."""
    modified_data = numpy_sample.copy()
    sat_key = SatelliteSampleKey.satellite_actual
    if sat_key in modified_data:
        modified_data.pop(sat_key)
    else:
        pytest.fail(f"Fixture 'numpy_sample' did not contain the key to be removed: {sat_key}")

    sample = UKRegionalSample(modified_data)
    expected_error_pattern = f"^Configuration expects Satellite data \\('{sat_key}'\\).*missing"

    with pytest.raises(ValueError, match=expected_error_pattern):
        sample.validate_sample(pvnet_configuration_object)


def test_validate_sample_with_wrong_shapes(numpy_sample, pvnet_configuration_object: Configuration):
    """Test validation raises ValueError when data shape is incorrect (GSP)."""
    modified_data = numpy_sample.copy()
    modified_data[GSPSampleKey.gsp] = np.random.rand(10)

    sample = UKRegionalSample(modified_data)

    # Depends on GSP shape calculated from pvnet_configuration_object
    with pytest.raises(ValueError, match="GSP shape mismatch at dimension 0"):
        sample.validate_sample(pvnet_configuration_object)
