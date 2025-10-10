"""
Base class testing - SampleBase
"""

import logging
import tempfile

import numpy as np
import pytest
import torch

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
    SiteSampleKey,
)
from ocf_data_sampler.torch_datasets.sample.base import (
    Sample,
    batch_to_tensor,
    copy_batch_to_device,
)


def test_sample_base_save_load(tmp_path):
    """Test basic save and load functionality"""

    sample = Sample(data={})
    sample._data["test_data"] = [1, 2, 3]

    save_path = tmp_path / "test_sample.dat"
    sample.save(save_path)
    assert save_path.exists()

    loaded_sample = Sample.load(save_path)
    assert isinstance(loaded_sample, Sample)


def test_batch_to_tensor_nested():
    """Test nested dictionary conversion"""
    batch = {"outer": {"inner": np.array([1, 2, 3])}}
    tensor_batch = batch_to_tensor(batch)

    assert torch.equal(tensor_batch["outer"]["inner"], torch.tensor([1, 2, 3]))


def test_batch_to_tensor_mixed_types():
    """Test handling of mixed data types"""
    batch = {
        "tensor_data": np.array([1, 2, 3]),
        "string_data": "not_a_tensor",
        "nested": {"numbers": np.array([4, 5, 6]), "text": "still_not_a_tensor"},
    }
    tensor_batch = batch_to_tensor(batch)

    assert isinstance(tensor_batch["tensor_data"], torch.Tensor)
    assert isinstance(tensor_batch["string_data"], str)
    assert isinstance(tensor_batch["nested"]["numbers"], torch.Tensor)
    assert isinstance(tensor_batch["nested"]["text"], str)


def test_batch_to_tensor_different_dtypes():
    """Test conversion of arrays with different dtypes"""
    batch = {
        "float_data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "int_data": np.array([1, 2, 3], dtype=np.int64),
        "bool_data": np.array([True, False, True], dtype=np.bool_),
    }
    tensor_batch = batch_to_tensor(batch)

    assert isinstance(tensor_batch["bool_data"], torch.Tensor)
    assert tensor_batch["float_data"].dtype == torch.float32
    assert tensor_batch["int_data"].dtype == torch.int64
    assert tensor_batch["bool_data"].dtype == torch.bool


def test_batch_to_tensor_multidimensional():
    """Test conversion of multidimensional arrays"""
    batch = {
        "matrix": np.array([[1, 2], [3, 4]]),
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    }
    tensor_batch = batch_to_tensor(batch)

    assert tensor_batch["matrix"].shape == (2, 2)
    assert tensor_batch["tensor"].shape == (2, 2, 2)
    assert torch.equal(tensor_batch["matrix"], torch.tensor([[1, 2], [3, 4]]))


def test_copy_batch_to_device():
    """Test moving tensors to a different device"""
    device = torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
    batch = {
        "tensor_data": torch.tensor([1, 2, 3]),
        "nested": {"matrix": torch.tensor([[1, 2], [3, 4]])},
        "non_tensor": "unchanged",
    }
    moved_batch = copy_batch_to_device(batch, device)

    assert moved_batch["tensor_data"].device == device
    assert moved_batch["nested"]["matrix"].device == device
    assert moved_batch["non_tensor"] == "unchanged"  # Non-tensors should remain unchanged


def test_site_sample_with_data(numpy_sample_site):
    """Testing of defined sample with actual data"""
    sample = Sample(numpy_sample_site)

    assert isinstance(sample._data, dict)
    assert sample._data["satellite_actual"].shape == (7, 1, 2, 2)
    assert sample._data["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    assert sample._data["site"].shape == (7,)
    assert sample._data["solar_azimuth"].shape == (7,)
    assert sample._data["date_sin"].shape == (7,)


def test_sample_save_load(numpy_sample_site):
    sample = Sample(numpy_sample_site)

    with tempfile.NamedTemporaryFile(suffix=".pt") as tf:
        sample.save(tf.name)
        loaded = Sample.load(tf.name)

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
    sample = Sample(numpy_sample_site)
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

def test_load_corrupted_file():
    """Test loading - corrupted / empty file"""
    with tempfile.NamedTemporaryFile(suffix=".pt") as tf, open(tf.name, "wb") as f:
        f.write(b"corrupted data")
        with pytest.raises(EOFError):
            Sample.load(tf.name)



def test_validate_sample(numpy_sample_gsp, pvnet_configuration_object: Configuration, caplog):
    """Test the validate_sample method succeeds with no warnings for a valid sample."""
    sample = Sample(numpy_sample_gsp)
    with caplog.at_level(logging.WARNING):
        result = sample.validate_sample(pvnet_configuration_object)

    assert isinstance(result, dict)
    assert result["valid"] is True
    assert len(caplog.records) == 0, "No warnings should be logged for a valid sample"


def test_validate_sample_with_missing_keys(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
):
    """Test validation raises ValueError when configured satellite data is missing."""
    modified_data = numpy_sample_gsp.copy()
    sat_key = SatelliteSampleKey.satellite_actual
    if sat_key in modified_data:
        modified_data.pop(sat_key)
    else:
        pytest.fail(f"Fixture 'numpy_sample_gsp' did not contain the key to be removed: {sat_key}")

    sample = Sample(modified_data)
    expected_error_pattern = f"^Configuration expects Satellite data \\('{sat_key}'\\).*missing"

    with pytest.raises(ValueError, match=expected_error_pattern):
        sample.validate_sample(pvnet_configuration_object)


def test_validate_sample_with_wrong_shapes(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
):
    """Test validation raises ValueError when data shape is incorrect (GSP)."""
    modified_data = numpy_sample_gsp.copy()
    modified_data[GSPSampleKey.gsp] = np.random.rand(10)

    sample = Sample(modified_data)

    with pytest.raises(ValueError, match="'GSP' shape mismatch: Actual shape:"):
        sample.validate_sample(pvnet_configuration_object)


def test_validate_sample_with_missing_solar_coors(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
):
    """Test validation raises ValueError when solar data is missing."""
    modified_data = numpy_sample_gsp.copy()
    solar_key = "solar_azimuth"
    modified_data.pop(solar_key)

    sample = Sample(modified_data)
    expected_error_pattern = f"^Configuration expects {solar_key} data but is missing"

    with pytest.raises(ValueError, match=expected_error_pattern):
        sample.validate_sample(pvnet_configuration_object)


def test_validate_sample_with_wrong_solar_shapes(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
):
    """Test validation raises ValueError when solar data shape is incorrect."""
    modified_data = numpy_sample_gsp.copy()
    modified_data["solar_azimuth"] = np.random.rand(10)

    sample = Sample(modified_data)

    with pytest.raises(ValueError, match="'Solar Azimuth data' shape mismatch: Actual shape:"):
        sample.validate_sample(pvnet_configuration_object)


def test_validate_sample_with_unexpected_provider(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
    caplog,
):
    """Test validation passes and logs a warning for an unexpected NWP provider."""
    modified_data = numpy_sample_gsp.copy()
    unexpected_provider = "unexpected_provider"
    nwp_data = {
        "nwp": np.random.rand(4, 1, 2, 2).astype(np.float32),
        "x": np.array([1, 2], dtype=np.int32),
        "y": np.array([1, 2], dtype=np.int32),
        NWPSampleKey.channel_names: ["t"],
    }
    if NWPSampleKey.nwp not in modified_data:
        modified_data[NWPSampleKey.nwp] = {}
    modified_data[NWPSampleKey.nwp][unexpected_provider] = nwp_data

    sample = Sample(modified_data)

    with caplog.at_level(logging.WARNING):
        result = sample.validate_sample(pvnet_configuration_object)

    # Validation should still pass
    assert isinstance(result, dict)
    assert result["valid"] is True

    warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_logs) == 1, "Expected exactly one warning log"

    log_message = warning_logs[0].message
    assert "Unexpected NWP providers found" in log_message
    assert unexpected_provider in log_message


def test_validate_sample_with_unexpected_component(
    numpy_sample_gsp,
    pvnet_configuration_object: Configuration,
    caplog,
):
    """Test validation passes and logs a warning for an unexpected component."""
    modified_data = numpy_sample_gsp.copy()
    unexpected_key = "unexpected_component_key_xyz"
    modified_data[unexpected_key] = np.random.rand(7).astype(np.float32)

    sample = Sample(modified_data)

    with caplog.at_level(logging.WARNING):
        result = sample.validate_sample(pvnet_configuration_object)

    # Validation should still pass
    assert isinstance(result, dict), "validate_sample should return a dictionary"
    assert result["valid"] is True, "Validation should pass even with warnings"

    warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_logs) == 1, "Expected exactly one warning log"

    log_message = warning_logs[0].message
    assert f"Unexpected component '{unexpected_key}'" in log_message
