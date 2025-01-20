# test_base.py

"""
Base class testing - SampleBase
"""

import pytest
from pathlib import Path
from ocf_data_sampler.sample.base import SampleBase


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
        with open(path, 'wb') as f:
            f.write(b'test_data')

    @classmethod
    def load(cls, path):
        """ Minimal load implementation """
        path = Path(path)
        instance = cls()
        return instance


def test_sample_base_initialization():
    """ Test initialization of SampleBase subclass """
    sample = TestSample()
    assert sample._data == {}, "Sample should start with empty dict"


def test_sample_base_getitem_setitem():
    """ Test basic get/set functionality """
    sample = TestSample()

    # Test with simple data types
    sample['int_data'] = 42
    assert sample['int_data'] == 42

    sample['list_data'] = [1, 2, 3]
    assert sample['list_data'] == [1, 2, 3]

    # Test with dictionary
    nested_data = {'a': 1, 'b': 2}
    sample['nested_data'] = nested_data
    assert sample['nested_data'] == nested_data

    # Test key error
    with pytest.raises(KeyError):
        _ = sample['nonexistent_key']


def test_sample_base_keys():
    """ Test keys method """
    sample = TestSample()
    
    sample['data1'] = 1
    sample['data2'] = 2

    assert set(sample.keys()) == {'data1', 'data2'}


def test_sample_base_save_load(tmp_path):
    """ Test basic save and load functionality """
    sample = TestSample()
    sample['test_data'] = [1, 2, 3]

    save_path = tmp_path / 'test_sample.dat'
    sample.save(save_path)
    assert save_path.exists()

    loaded_sample = TestSample.load(save_path)
    assert isinstance(loaded_sample, TestSample)


def test_sample_base_abstract_methods():
    """ Test abstract method enforcement """
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SampleBase()


def test_data_operations():
    """ Test various data operations """
    sample = TestSample()
    
    # Test multiple set operations
    test_data = {
        'string': 'test',
        'number': 123,
        'list': [1, 2, 3],
        'dict': {'a': 1, 'b': 2}
    }
    
    for key, value in test_data.items():
        sample[key] = value
        
    # Verify all data was stored correctly
    for key, value in test_data.items():
        assert sample[key] == value
        
    # Test keys match expected
    assert set(sample.keys()) == set(test_data.keys())
