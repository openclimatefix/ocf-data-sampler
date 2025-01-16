# test_base.py

"""
Base class testing - SampleBase
"""

import pytest
import numpy as np
import torch
import xarray as xr
from pathlib import Path

from ocf_data_sampler.sample.base import SampleBase


# Base class define
class TestSample(SampleBase):
    """
    SampleBase for testing purposes
    Minimal implementations - abstract methods
    """

    def plot(self, **kwargs):
        """ Minimal plot implementation """
        return None

    def validate(self) -> None:
        """ Minimal validation implementation """
        pass

    def save(self, path):
        """ Minimal save implementation """

        path = Path(path)
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {path.suffix}")
        with open(path, 'wb') as f:
            f.write(b'test_data')

    @classmethod
    def load(cls, path):
        """ Minimal load implementation """

        path = Path(path)
        if path.suffix not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format {path.suffix}")
        
        instance = cls()
        return instance


def test_sample_base_initialization():
    """ Initialisation of SampleBase subclass """

    sample = TestSample()
    assert sample._data == {}, "Sample should start with empty dict"


def test_sample_base_getitem_setitem():
    """ Get / set items - type validation"""

    sample = TestSample()

    # Get / set np array
    arr = np.array([1, 2, 3])
    sample['numpy_data'] = arr
    assert np.array_equal(sample['numpy_data'], arr)

    # Get / set tensor
    tensor = torch.tensor([4, 5, 6])
    sample['torch_data'] = tensor
    assert torch.equal(sample['torch_data'], tensor)

    # Get / set xr DataArray
    xr_data = xr.DataArray([7, 8, 9])
    sample['xarray_data'] = xr_data

    assert sample['xarray_data'].identical(xr_data)

    # Setting nested dict
    nested_data = {
        'sub_numpy': np.array([10, 11, 12]),
        'sub_torch': torch.tensor([13, 14, 15])
    }
    sample['nested_data'] = nested_data
    assert 'nested_data' in sample.keys()
    assert np.array_equal(sample['nested_data']['sub_numpy'], nested_data['sub_numpy'])
    assert torch.equal(sample['nested_data']['sub_torch'], nested_data['sub_torch'])


def test_sample_base_value_validation():
    """ Test value validation mechanism """

    sample = TestSample()

    # Valid inputs - pass
    valid_inputs = [
        np.array([1, 2, 3]),
        torch.tensor([4, 5, 6]),
        xr.DataArray([7, 8, 9]),
        None,
        {'nested': np.array([10, 11, 12])}
    ]

    for input_value in valid_inputs:
        try:
            sample['valid_data'] = input_value
        except Exception as e:
            pytest.fail(f"Valid input {input_value} raised exception: {e}")

    # Invalid inputs - TypeError
    invalid_inputs = [
        [1, 2, 3],
        "string",
        {'invalid': [1, 2, 3]}
    ]

    for input_value in invalid_inputs:
        with pytest.raises(TypeError, match="Value must be"):
            sample['invalid_data'] = input_value


def test_sample_base_keys():
    """ Keys method """

    sample = TestSample()
    sample['data1'] = np.array([1, 2, 3])
    sample['data2'] = torch.tensor([4, 5, 6])

    assert set(sample.keys()) == {'data1', 'data2'}


def test_sample_base_save_load(tmp_path):
    """ Save and load methods """

    sample = TestSample()
    sample['test_data'] = np.array([1, 2, 3])

    save_path = tmp_path / 'test_sample.pt'
    sample.save(save_path)
    assert save_path.exists()

    loaded_sample = TestSample.load(save_path)
    assert isinstance(loaded_sample, TestSample)

    # ValueError checking
    unsupported_path = tmp_path / 'test_sample.txt'
    with pytest.raises(ValueError, match="Unsupported format"):
        sample.save(unsupported_path)
    
    with pytest.raises(ValueError, match="Unsupported format"):
        TestSample.load(unsupported_path)


def test_sample_base_abstract_methods():
    """ Test abstract method instantiation """

    with pytest.raises(TypeError, match="Can't instantiate abstract class SampleBase without an implementation for abstract methods"):
        SampleBase()
