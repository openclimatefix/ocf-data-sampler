# test_utils.py

"""
Utils testing
"""

import pytest
import numpy as np
from typing import Dict, Any

from ocf_data_sampler.sample.base import SampleBase

from ocf_data_sampler.sample.utils import (
    stack_samples,
    merge_samples,
    convert_batch_to_sample
)


# Dummy class define
class TestSample(SampleBase):
    def plot(self, **kwargs):
        pass
        
    def validate(self) -> None:
        pass
        
    def save(self, path: str) -> None:
        pass
        
    @classmethod
    def load(cls, path: str) -> 'TestSample':
        return cls()


def create_test_sample(data: Dict[str, Any]) -> TestSample:
    """ Create test sample """

    sample = TestSample()
    for key, value in data.items():
        sample[key] = value
    return sample


@pytest.fixture
def simple_samples():
    """ Fixture - 'simple' test samples """

    sample1 = create_test_sample({
        'data1': np.array([1, 2, 3]),
        'data2': np.array([4, 5, 6])
    })
    sample2 = create_test_sample({
        'data1': np.array([7, 8, 9]),
        'data2': np.array([10, 11, 12])
    })
    return [sample1, sample2]


@pytest.fixture
def nested_samples():
    """ Fixture - nested test samples """

    sample1 = create_test_sample({
        'nwp': {
            'ukv': np.array([[1, 2], [3, 4]]),
            'ecmwf': np.array([[5, 6], [7, 8]])
        },
        'data': np.array([1, 2, 3])
    })
    sample2 = create_test_sample({
        'nwp': {
            'ukv': np.array([[9, 10], [11, 12]]),
            'ecmwf': np.array([[13, 14], [15, 16]])
        },
        'data': np.array([4, 5, 6])
    })
    return [sample1, sample2]


def test_stack_samples_simple(simple_samples):
    """ Test stacking 'simple' samples """

    stacked = stack_samples(simple_samples)
    
    assert isinstance(stacked, TestSample)
    assert 'data1' in stacked.keys()
    assert 'data2' in stacked.keys()
    
    np.testing.assert_array_equal(
        stacked['data1'],
        np.stack([s['data1'] for s in simple_samples])
    )
    np.testing.assert_array_equal(
        stacked['data2'],
        np.stack([s['data2'] for s in simple_samples])
    )


def test_stack_samples_nested(nested_samples):
    """ Test stacking nested samples """

    stacked = stack_samples(nested_samples)
    
    assert isinstance(stacked, TestSample)
    assert 'nwp' in stacked.keys()
    assert 'data' in stacked.keys() 
    assert 'ukv' in stacked['nwp']
    assert 'ecmwf' in stacked['nwp']
    
    np.testing.assert_array_equal(
        stacked['nwp']['ukv'],
        np.stack([s['nwp']['ukv'] for s in nested_samples])
    )
    
    np.testing.assert_array_equal(
        stacked['data'],
        np.stack([s['data'] for s in nested_samples])
    )


def test_stack_samples_empty():
    """ Test stacking empty sample list """

    with pytest.raises(ValueError, match="Cannot stack empty list of samples"):
        stack_samples([])


def test_stack_samples_different_types():
    """ Test stacking samples of different types """

    # 'Other' dummy sample define
    class OtherSample(SampleBase):
        def plot(self, **kwargs):
            pass
            
        def validate(self) -> None:
            pass
            
        def save(self, path: str) -> None:
            pass
            
        @classmethod
        def load(cls, path: str) -> 'OtherSample':
            return cls()
            
    sample1 = TestSample()
    sample2 = OtherSample()
    
    with pytest.raises(TypeError, match="All samples must be of same type"):
        stack_samples([sample1, sample2])


def test_merge_samples(simple_samples):
    """ Sample merge testing """

    simple_samples[1]['data3'] = np.array([13, 14, 15])
    merged = merge_samples(simple_samples)
    
    assert isinstance(merged, TestSample)
    assert set(merged.keys()) == {'data1', 'data2', 'data3'}
    np.testing.assert_array_equal(merged['data1'], simple_samples[1]['data1'])


def test_merge_samples_empty():
    """ Test merging empty sample list """

    with pytest.raises(ValueError, match="Cannot merge empty list of samples"):
        merge_samples([])


def test_merge_samples_warning(simple_samples, caplog):
    """ Test warning when merging samples with duplicate keys """

    merged = merge_samples(simple_samples)
    assert any("Key data1 already exists in merged sample" in record.message 
              for record in caplog.records)


def test_convert_batch_to_sample_simple():
    """ Test converting simple batch dict to sample """

    batch_dict = {
        'data1': np.array([1, 2, 3]),
        'data2': np.array([4, 5, 6])
    }
    
    sample = convert_batch_to_sample(batch_dict, TestSample)
    
    assert isinstance(sample, TestSample)
    assert set(sample.keys()) == {'data1', 'data2'}
    np.testing.assert_array_equal(sample['data1'], batch_dict['data1'])


def test_convert_batch_to_sample_nested():
    """ Test converting nested batch dict to sample """

    batch_dict = {
        'nwp': {
            'ukv': np.array([[1, 2], [3, 4]]),
            'ecmwf': np.array([[5, 6], [7, 8]])
        },
        'data': np.array([1, 2, 3])
    }
    
    sample = convert_batch_to_sample(batch_dict, TestSample)
    
    assert isinstance(sample, TestSample)
    assert 'nwp' in sample.keys()
    assert set(sample['nwp'].keys()) == {'ukv', 'ecmwf'}
    np.testing.assert_array_equal(sample['nwp']['ukv'], batch_dict['nwp']['ukv'])


def test_convert_batch_to_sample_empty():
    """ Test converting empty batch dict to sample """
    
    sample = convert_batch_to_sample({}, TestSample)
    assert isinstance(sample, TestSample)
    assert len(sample.keys()) == 0