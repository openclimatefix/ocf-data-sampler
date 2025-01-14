# test_base.py

"""
Base class testing - SampleBase
Logging included purely for non-suppressed execution
"""

import numpy as np
import torch
import xarray as xr
import pytest
import tempfile
import logging

from pathlib import Path

from ocf_data_sampler.sample.base import SampleBase


logger = logging.getLogger(__name__)


class SimpleBatch(SampleBase):
    """ Test implementation - SampleBase """
    
    # Replication to ensure inheritance
    @classmethod
    def load(cls, path):
        path = Path(path)
        instance = cls()
        try:
            if path.suffix == '.nc':
                ds = xr.open_dataset(path)
                instance._data = cls._dataset_to_dict(ds)
                ds.close()
            elif path.suffix == '.zarr':
                ds = xr.open_zarr(path)
                instance._data = cls._dataset_to_dict(ds)
                ds.close()
            else:
                with np.load(path, allow_pickle=True) as data:
                    loaded_data = {}
                    for key in data.files:
                        if '/' in key:
                            main_key, sub_key = key.split('/')
                            if main_key not in loaded_data:
                                loaded_data[main_key] = {}
                            loaded_data[main_key][sub_key] = data[key]
                        else:
                            loaded_data[key] = data[key]
                    instance._data = loaded_data
        except Exception as e:
            raise
        return instance

    def plot(self, **kwargs):
        pass


@pytest.fixture
def simple_batch():
    """Fixture - 'simple' batch instance """
    return SimpleBatch()


@pytest.fixture
def nested_batch():
    """Fixture - batch with nested data """
    batch = SimpleBatch()
    batch['flat'] = np.array([1, 2, 3])
    batch['nested'] = {
        'a': np.array([4, 5, 6]),
        'b': np.array([7, 8, 9])
    }
    return batch


@pytest.fixture
def temp_dir():
    """Fixture - temp directory """
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


def test_batch_getitem_setitem():
    """ Test basic functionality """
    batch = SimpleBatch()
    batch["test"] = np.array([1, 2, 3])

    assert np.array_equal(batch["test"], np.array([1, 2, 3]))
    logger.info("Assertion - batch['test'] equals expected array")

    with pytest.raises(KeyError):
        _ = batch["nonexistent"]
    logger.info("Assertion - error raised for nonexistent key")


def test_batch_to_torch():
    """ Test conversion to tensors """
    batch = SimpleBatch()
    batch["data"] = np.array([1.0, 2.0, 3.0])
    batch.to_torch()
    assert isinstance(batch["data"], torch.Tensor)
    logger.info("Assertion - batch['data'] is tensor")


def test_batch_to_numpy():
    """ Test conversion to numpy array """
    batch = SimpleBatch()
    batch["data"] = torch.tensor([1.0, 2.0, 3.0])
    batch.to_numpy()
    assert isinstance(batch["data"], np.ndarray)
    logger.info("Assertion - batch['data'] is numpy array")


def test_nested_conversions():
    """ Test conversion - nested data """
    batch = SimpleBatch()
    batch["nested"] = {
        "a": np.array([1.0]),
        "b": torch.tensor([2.0])
    }
    batch.to_torch()
    assert isinstance(batch["nested"]["a"], torch.Tensor)
    assert isinstance(batch["nested"]["b"], torch.Tensor)
    batch.to_numpy()
    assert isinstance(batch["nested"]["a"], np.ndarray)
    assert isinstance(batch["nested"]["b"], np.ndarray)


def test_xarray_conversions():
    """ Test conversion with xarray """
    batch = SimpleBatch()
    batch["data"] = xr.DataArray([1.0, 2.0, 3.0])
    batch.to_torch()
    assert isinstance(batch["data"], torch.Tensor)
    batch.to_numpy()
    assert isinstance(batch["data"], np.ndarray)


@pytest.mark.parametrize("fmt", [".npz", ".nc", ".zarr"])
def test_save_load(temp_dir, fmt):
    """ Test save / load with different format types """
    batch = SimpleBatch()
    batch["data"] = np.array([1.0, 2.0, 3.0])
    batch["nested"] = {"a": np.array([4.0, 5.0, 6.0])}
    
    path = temp_dir / f"test{fmt}"
    batch.save(path)
    loaded = SimpleBatch.load(path)
    
    assert np.array_equal(loaded["data"], batch["data"])
    logger.info(f"Assertion - batch saved / loaded successfully - {fmt}")

    assert np.array_equal(loaded["nested"]["a"], batch["nested"]["a"])


def test_save_load_with_nans(temp_dir):
    """ Test save / load with NaN values """
    batch = SimpleBatch()
    batch["data"] = np.array([1.0, np.nan, 3.0])
    path = temp_dir / "test.npz"
    batch.save(path)
    loaded = SimpleBatch.load(path)
    assert np.array_equal(loaded["data"], batch["data"], equal_nan=True)


def test_fill_nans():
    """ Test NaN imputation """
    batch = SimpleBatch()
    batch["data"] = np.array([1.0, np.nan, 3.0])
    batch.fill_nans()
    assert not np.isnan(batch["data"]).any()
    assert np.array_equal(batch["data"], np.array([1.0, 0.0, 3.0]))


def test_mixed_precision():
    """ Test handling mixed precision """
    batch = SimpleBatch()
    batch["float32"] = np.array([1.0, 2.0], dtype=np.float32)
    batch["float64"] = np.array([1.0, 2.0], dtype=np.float64)
    batch.to_torch()
    assert batch["float32"].dtype == torch.float32
    assert batch["float64"].dtype == torch.float64


def test_dimension_preservation():
    """ Test preservation """
    batch = SimpleBatch()
    shapes = {
        "1d": (10,),
        "2d": (5, 5),
        "3d": (2, 3, 4)
    }
    for name, shape in shapes.items():
        batch[name] = np.zeros(shape)
    
    batch.to_torch()
    batch.to_numpy()
    
    for name, shape in shapes.items():
        assert batch[name].shape == shape

# For both valid and invalid types
# Following two tests
def test_value_validation():
    """ Test value validation """
    batch = SimpleBatch()
    batch["numpy"] = np.array([1])
    batch["torch"] = torch.tensor([1])
    batch["xarray"] = xr.DataArray([1])

    with pytest.raises(TypeError):
        batch["invalid"] = [1, 2, 3]


def test_nested_value_validation():
    """ Test nested value validation """
    batch = SimpleBatch()
    batch["valid"] = {
        "a": np.array([1]),
        "b": torch.tensor([2])
    }
    
    with pytest.raises(TypeError):
        batch["invalid"] = {
            "a": np.array([1]),
            "b": [1, 2, 3]
        }

############################

class MockSample(SampleBase):
    """Mock implementation of SampleBase for testing purposes."""
    def plot(self, **kwargs) -> None:
        pass  # No-op implementation for the abstract method

def test_to_torch_nested():
    """Test to_torch for nested structures."""
    sample = MockSample()
    sample['nwp'] = {'ukv': {'nwp': np.random.rand(5, 5)}}
    sample.to_torch()
    assert isinstance(sample['nwp']['ukv']['nwp'], torch.Tensor)
