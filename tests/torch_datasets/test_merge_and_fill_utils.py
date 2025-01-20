import numpy as np

from ocf_data_sampler.torch_datasets.utils.merge_and_fill_utils import (
    merge_dicts,
    fill_nans_in_arrays,
)

def test_merge_dicts():
    """Test merge_dicts function"""
    dict1 = {"a": 1, "b": 2}
    dict2 = {"c": 3, "d": 4}
    dict3 = {"e": 5}
    
    result = merge_dicts([dict1, dict2, dict3])
    assert result == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    
    # Test key overwriting
    dict4 = {"a": 10, "f": 6}
    result = merge_dicts([dict1, dict4])
    assert result["a"] == 10


def test_fill_nans_in_arrays():
    """Test the fill_nans_in_arrays function"""
    array_with_nans = np.array([1.0, np.nan, 3.0, np.nan])
    nested_dict = {
        "array1": array_with_nans,
        "nested": {
            "array2": np.array([np.nan, 2.0, np.nan, 4.0])
        },
        "string_key": "not_an_array"
    }
    
    result = fill_nans_in_arrays(nested_dict)
    
    assert not np.isnan(result["array1"]).any()
    assert np.array_equal(result["array1"], np.array([1.0, 0.0, 3.0, 0.0]))
    assert not np.isnan(result["nested"]["array2"]).any()
    assert np.array_equal(result["nested"]["array2"], np.array([0.0, 2.0, 0.0, 4.0]))
    assert result["string_key"] == "not_an_array"


