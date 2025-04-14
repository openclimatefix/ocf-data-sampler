"""
UK Regional class testing - UKRegionalSample
"""

import tempfile

import numpy as np
import pytest

from ocf_data_sampler.numpy_sample import GSPSampleKey, NWPSampleKey, SatelliteSampleKey
from ocf_data_sampler.torch_datasets.sample.uk_regional import UKRegionalSample


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
        "nwp": np.random.rand(4, 1, 3, 3),
        "x": np.array([1, 2, 3]),
        "y": np.array([1, 2, 3]),
        NWPSampleKey.channel_names: ["test_channel"],
    }
    return {
        "nwp": {
            "ukv": nwp_data,
        },
        GSPSampleKey.gsp: np.random.rand(7),
        SatelliteSampleKey.satellite_actual: np.random.rand(7, 1, 2, 2),
    }


@pytest.fixture
def custom_configuration():
    """Custom configuration for testing"""
    return {
        "required_keys": [GSPSampleKey.gsp, NWPSampleKey.nwp],
        "expected_shapes": {
            GSPSampleKey.gsp: (7,),
            NWPSampleKey.nwp: (4, 1, 3, 3),
            SatelliteSampleKey.satellite_actual: (7, 1, 2, 2),
        },
    }


@pytest.fixture
def shape_configuration():
    """Configuration with expected shapes for testing"""
    return {
        "expected_shapes": {
            GSPSampleKey.gsp: (7,),
        },
    }


@pytest.fixture
def input_data_configuration():
    """Config for testing shape calculation"""
    return {
        "required_keys": [GSPSampleKey.gsp, NWPSampleKey.nwp],
        "input_data": {
            "gsp": {
                "time_resolution_minutes": 30,
                "interval_start_minutes": -180,
                "interval_end_minutes": 0,
            },
            "nwp": {
                "ukv": {
                    "image_size_pixels_height": 3,
                    "image_size_pixels_width": 3,
                },
            },
            "satellite": {
                "channels": ["HRV"],
                "time_resolution_minutes": 30,
                "interval_start_minutes": -180,
                "interval_end_minutes": 0,
                "image_size_pixels_height": 2,
                "image_size_pixels_width": 2,
            },
        },
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


def test_validate_sample(numpy_sample, custom_configuration):
    """Test the validate_sample method with config"""
    sample = UKRegionalSample(numpy_sample)

    # Assert validation with provided config
    result = sample.validate_sample(custom_configuration)
    assert result is True


def test_validate_sample_with_custom_config(numpy_sample, custom_configuration):
    """Test the validate_sample method with custom config"""

    sample = UKRegionalSample(numpy_sample)

    # Validate with the custom config
    result = sample.validate_sample(custom_configuration)
    assert result is True


def test_validate_sample_with_input_data_config(numpy_sample, input_data_configuration):
    """Test the validate_sample method with input_data configuration"""
    sample = UKRegionalSample(numpy_sample)

    # Validate with the input_data config
    result = sample.validate_sample(input_data_configuration)
    assert result is True


def test_validate_sample_with_missing_keys(numpy_sample, custom_configuration):
    """Test validation with missing required keys"""
    # Make a copy to avoid modifying fixture
    modified_data = numpy_sample.copy()

    # Remove satellite data using pop()
    modified_data.pop(SatelliteSampleKey.satellite_actual)
    sample = UKRegionalSample(modified_data)

    # Use config that requires satellite data
    config_with_satellite = custom_configuration.copy()
    config_with_satellite["required_keys"] = [
        GSPSampleKey.gsp,
        NWPSampleKey.nwp,
        SatelliteSampleKey.satellite_actual,
    ]

    with pytest.raises(ValueError, match="Missing required key: satellite_actual"):
        sample.validate_sample(config_with_satellite)


def test_validate_sample_with_wrong_shapes(numpy_sample, shape_configuration):
    """Test validation with incorrect data shapes"""
    # Create copy of sample with wrong GSP shape
    modified_data = numpy_sample.copy()
    modified_data[GSPSampleKey.gsp] = np.random.rand(10)

    sample = UKRegionalSample(modified_data)

    with pytest.raises(ValueError, match="GSP shape mismatch at dimension 0"):
        sample.validate_sample(shape_configuration)
