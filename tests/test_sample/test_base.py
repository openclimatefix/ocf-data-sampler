"""
Base class testing - SampleBase
"""

import pytest
import torch
import numpy as np

from pathlib import Path
from ocf_data_sampler.sample.base import (
    SampleBase,
    batch_to_tensor,
    copy_batch_to_device
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


def test_batch_to_tensor_nested():
    """ Test nested dictionary conversion """
    batch = {
        'outer': {
            'inner': np.array([1, 2, 3])
        }
    }
    tensor_batch = batch_to_tensor(batch)
    
    assert torch.equal(tensor_batch['outer']['inner'], torch.tensor([1, 2, 3]))


def test_batch_to_tensor_mixed_types():
    """ Test handling of mixed data types """
    batch = {
        'tensor_data': np.array([1, 2, 3]),
        'string_data': 'not_a_tensor',
        'nested': {
            'numbers': np.array([4, 5, 6]),
            'text': 'still_not_a_tensor'
        }
    }
    tensor_batch = batch_to_tensor(batch)
    
    assert isinstance(tensor_batch['tensor_data'], torch.Tensor)
    assert isinstance(tensor_batch['string_data'], str)
    assert isinstance(tensor_batch['nested']['numbers'], torch.Tensor)
    assert isinstance(tensor_batch['nested']['text'], str)


def test_batch_to_tensor_different_dtypes():
    """ Test conversion of arrays with different dtypes """
    batch = {
        'float_data': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        'int_data': np.array([1, 2, 3], dtype=np.int64),
        'bool_data': np.array([True, False, True], dtype=np.bool_)
    }
    tensor_batch = batch_to_tensor(batch)
    
    assert isinstance(tensor_batch['bool_data'], torch.Tensor)
    assert tensor_batch['float_data'].dtype == torch.float32
    assert tensor_batch['int_data'].dtype == torch.int64
    assert tensor_batch['bool_data'].dtype == torch.bool


def test_batch_to_tensor_multidimensional():
    """ Test conversion of multidimensional arrays """
    batch = {
        'matrix': np.array([[1, 2], [3, 4]]),
        'tensor': np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    }
    tensor_batch = batch_to_tensor(batch)
    
    assert tensor_batch['matrix'].shape == (2, 2)
    assert tensor_batch['tensor'].shape == (2, 2, 2)
    assert torch.equal(tensor_batch['matrix'], torch.tensor([[1, 2], [3, 4]]))


def test_copy_batch_to_device():
    """ Test moving tensors to a different device """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {
        'tensor_data': torch.tensor([1, 2, 3]),
        'nested': {
            'matrix': torch.tensor([[1, 2], [3, 4]])
        },
        'non_tensor': 'unchanged'
    }
    moved_batch = copy_batch_to_device(batch, device)
    assert moved_batch['tensor_data'].device == device
    assert moved_batch['nested']['matrix'].device == device
    assert moved_batch['non_tensor'] == 'unchanged'  # Non-tensors should remain unchanged