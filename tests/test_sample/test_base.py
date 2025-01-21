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


def test_sample_base_initialisation():
    """ Initialisation of SampleBase subclass """

    sample = TestSample()
    assert sample._data == {}, "Sample should start with empty dict"


def test_sample_base_keys():
    """ Test keys method """

    sample = TestSample()
    sample._data['data1'] = 1
    sample._data['data2'] = 2
    assert set(sample.keys()) == {'data1', 'data2'}


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


def test_sample_base_initialisation_with_data():
    """ Test initialisation of SampleBase with data """

    test_data = {
        'int_data': 42,
        'list_data': [1, 2, 3],
        'nested_data': {'a': 1, 'b': 2}
    }
    
    sample = TestSample(data=test_data)
    assert sample._data == test_data, "Sample initialised with provided data"
