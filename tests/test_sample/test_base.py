"""
Base class testing - SampleBase
"""

import pytest
import torch
import numpy as np

from pathlib import Path
from ocf_data_sampler.sample.base import (
    SampleBase, 
    NumpySample, 
    TensorSample
)

class TestSample(SampleBase):
    """
    SampleBase for testing purposes
    Minimal implementations - abstract methods
    """

    def __init__(self):
        super().__init__()
        self._data = {}

    def plot(self, **kwargs):
        """ Minimal plot implementation """
        return None

    def to_numpy(self) -> None:
        """ Standard implementation """
        return {key: np.array(value) for key, value in self._data.items()}

    def save(self, path):
        """ Minimal save implementation """
        path = Path(path)
        with open(path, 'wb') as f:
            f.write(b'test_data')

    @classmethod
    def load(cls, path):
        """ Minimal load implementation """
        path = Path(path)
        instance = cls()
        return instance


def test_sample_base_initialisation():
    """ Initialisation of SampleBase subclass """

    sample = TestSample()
    assert hasattr(sample, '_data'), "Sample should have _data attribute"
    assert sample._data == {}, "Sample should start with empty dict"


def test_sample_base_save_load(tmp_path):
    """ Test basic save and load functionality """

    sample = TestSample()
    sample._data['test_data'] = [1, 2, 3]

    save_path = tmp_path / 'test_sample.dat'
    sample.save(save_path)
    assert save_path.exists()

    loaded_sample = TestSample.load(save_path)
    assert isinstance(loaded_sample, TestSample)


def test_sample_base_abstract_methods():
    """ Test abstract method enforcement """
    
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SampleBase()


def test_sample_base_to_numpy():
    """ Test the to_numpy functionality """
    import numpy as np
    
    sample = TestSample()
    sample._data = {
        'int_data': 42,
        'list_data': [1, 2, 3]
    }
    numpy_data = sample.to_numpy()

    assert isinstance(numpy_data, dict)
    assert all(isinstance(value, np.ndarray) for value in numpy_data.values())
    assert np.array_equal(numpy_data['list_data'], np.array([1, 2, 3]))


def test_numpy_sample_initialisation():
    """ NumpySample init with / without data """
    sample = NumpySample()
    assert hasattr(sample, '_data')
    assert sample._data == {}
    data = {'test': np.array([1, 2, 3])}
    sample = NumpySample(data)
    assert sample._data == data


def test_numpy_sample_nested_structure():
    """ NumpySample with nested structure """
    nested_data = {
        'simple': np.array([1, 2, 3]),
        'nested': {
            'sub1': np.array([[1, 2], [3, 4]]),
            'sub2': np.array([5, 6])
        }
    }
    sample = NumpySample(nested_data)
    
    # Assert retrieval and setting
    assert np.array_equal(sample['simple'], nested_data['simple'])
    assert np.array_equal(sample['nested']['sub1'], nested_data['nested']['sub1'])    
    new_data = np.array([7, 8, 9])
    sample['simple'] = new_data
    assert np.array_equal(sample['simple'], new_data)


def test_numpy_sample_to_numpy():
    """ Test NumpySample to_numpy conversion """
    data = {
        'array1': np.array([1, 2, 3]),
        'array2': np.array([[4, 5], [6, 7]])
    }
    sample = NumpySample(data)

    numpy_data = sample.to_numpy()
    assert numpy_data == data
    assert all(isinstance(v, np.ndarray) for v in numpy_data.values())


def test_tensor_sample_initialisation():
    """ Test TensorSample init with / without data """
    sample = TensorSample()
    assert hasattr(sample, '_data')
    assert sample._data == {}
    data = {'test': torch.tensor([1, 2, 3])}
    sample = TensorSample(data)
    assert torch.equal(sample._data['test'], data['test'])


def test_tensor_sample_nested_structure():
    """ Test TensorSample with nested structure """
    nested_data = {
        'simple': torch.tensor([1, 2, 3]),
        'nested': {
            'sub1': torch.tensor([[1, 2], [3, 4]]),
            'sub2': torch.tensor([5, 6])
        }
    }
    sample = TensorSample(nested_data)
    
    # Assert retrieval and setting
    assert torch.equal(sample['simple'], nested_data['simple'])
    assert torch.equal(sample['nested']['sub1'], nested_data['nested']['sub1'])    
    new_data = torch.tensor([7, 8, 9])
    sample['simple'] = new_data
    assert torch.equal(sample['simple'], new_data)


def test_tensor_sample_to_numpy():
    """ Test TensorSample to_numpy conversion """
    data = {
        'tensor1': torch.tensor([1, 2, 3]),
        'tensor2': torch.tensor([[4, 5], [6, 7]]),
        'nested': {
            'sub': torch.tensor([8, 9])
        }
    }
    sample = TensorSample(data)
    numpy_data = sample.to_numpy()
    
    # Assert conversion to numpy and check values
    assert isinstance(numpy_data['tensor1'], np.ndarray)
    assert isinstance(numpy_data['tensor2'], np.ndarray)
    assert isinstance(numpy_data['nested']['sub'], np.ndarray)    
    assert np.array_equal(numpy_data['tensor1'], data['tensor1'].numpy())
    assert np.array_equal(numpy_data['tensor2'], data['tensor2'].numpy())
    assert np.array_equal(numpy_data['nested']['sub'], data['nested']['sub'].numpy())
