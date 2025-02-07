"""
Base class testing - SampleBase
"""

import pytest
import numpy as np

from pathlib import Path
from ocf_data_sampler.sample.base import SampleBase


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
