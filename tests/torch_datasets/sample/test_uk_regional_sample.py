"""
UK Regional class testing - UKRegionalSample
"""

import tempfile

import numpy as np
import pytest

from ocf_data_sampler.numpy_sample import GSPSampleKey, NWPSampleKey, SatelliteSampleKey
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample, validate_samples


@pytest.fixture
def pvnet_config_filename(tmp_path):
    """Minimal config file - testing"""
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


@pytest.fixture
def numpy_sample():
    """Synthetic data generation"""

    # Field / spatial coordinates
    nwp_data = {
        "nwp": np.random.rand(4, 1, 2, 2),
        "x": np.array([1, 2]),
        "y": np.array([1, 2]),
        NWPSampleKey.channel_names: ["test_channel"],
    }

    return {
        "nwp": {
            "ukv": nwp_data,
        },
        GSPSampleKey.gsp: np.random.rand(7),
        SatelliteSampleKey.satellite_actual: np.random.rand(7, 1, 2, 2),
    }


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

# Add these tests to the existing test_uk_regional_sample.py file

# Add this fixture to create a validation config file
@pytest.fixture
def validation_config_file(tmp_path):
    """Create a validation config file for testing"""
    config_content = """
    required_keys:
      - gsp
      - nwp
      - satellite_actual
    expected_shapes:
      gsp: [7]
    nwp_shape: [2, 2]
    satellite_shape: [2, 2]
    """
    config_file = tmp_path / "validation_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


def test_validate_sample(numpy_sample):
    """Test the validate_sample method with default config"""
    sample = UKRegionalSample(numpy_sample)
    validation_result = sample.validate_sample()

    # Check validation structure
    assert isinstance(validation_result, dict)
    assert "valid" in validation_result
    assert "errors" in validation_result

    # Should be valid for the default fixture data
    assert validation_result["valid"] is True
    assert len(validation_result["errors"]) == 0


def test_validate_sample_with_custom_config(numpy_sample):
    """Test the validate_sample method with custom config"""
    sample = UKRegionalSample(numpy_sample)

    # Create a custom config
    custom_config = {
        "required_keys": [GSPSampleKey.gsp, NWPSampleKey.nwp],
        "expected_shapes": {
            GSPSampleKey.gsp: (7,),
        },
        "nwp_shape": (2, 2),
    }

    validation_result = sample.validate_sample(custom_config)
    assert validation_result["valid"] is True
    assert len(validation_result["errors"]) == 0


def test_validate_sample_with_missing_keys(numpy_sample):
    """Test validation with missing required keys"""
    # Create a copy of the sample without satellite data
    modified_data = {
        "nwp": numpy_sample["nwp"],
        GSPSampleKey.gsp: numpy_sample[GSPSampleKey.gsp],
        # Satellite key intentionally removed
    }

    sample = UKRegionalSample(modified_data)

    # Create a config that requires satellite data
    config = {
        "required_keys": [
            GSPSampleKey.gsp,
            NWPSampleKey.nwp,
            SatelliteSampleKey.satellite_actual,
        ],
    }

    validation_result = sample.validate_sample(config)
    assert validation_result["valid"] is False
    assert any(
        "Missing required key: satellite_actual" in error
        for error in validation_result["errors"]
    )

def test_validate_sample_with_wrong_shapes(numpy_sample):
    """Test validation with incorrect data shapes"""
    # Create a copy of the sample with wrong GSP shape
    modified_data = numpy_sample.copy()
    modified_data[GSPSampleKey.gsp] = np.random.rand(10)

    sample = UKRegionalSample(modified_data)

    config = {
        "expected_shapes": {
            GSPSampleKey.gsp: (7,),
        },
    }

    validation_result = sample.validate_sample(config)
    assert validation_result["valid"] is False
    assert any("Shape mismatch for gsp" in error for error in validation_result["errors"])


def test_validate_samples_function(numpy_sample):
    """Test the validate_samples function for batch validation"""
    # Create one valid and one invalid sample
    valid_sample = UKRegionalSample(numpy_sample)

    # Create an invalid sample with wrong GSP shape
    modified_data = numpy_sample.copy()
    modified_data[GSPSampleKey.gsp] = np.random.rand(10)
    invalid_sample = UKRegionalSample(modified_data)

    # Test batch validation with in-memory config instead of file
    samples = [valid_sample, invalid_sample]
    config = {
        "required_keys": [
            GSPSampleKey.gsp,
            NWPSampleKey.nwp,
            SatelliteSampleKey.satellite_actual,
        ],
        "expected_shapes": {
            GSPSampleKey.gsp: (7,),
        },
    }

    # Pass the config directly instead of via file path
    results = validate_samples(samples, config)

    assert results["total_samples"] == 2
    assert results["valid_samples"] == 1
    assert results["invalid_samples"] == 1
    assert len(results["error_summary"]) > 0

    # Also test without a config
    results = validate_samples(samples)
    assert results["total_samples"] == 2
