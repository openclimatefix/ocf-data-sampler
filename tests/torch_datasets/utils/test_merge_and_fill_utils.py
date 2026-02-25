import numpy as np

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.torch_datasets.utils.merge_and_fill_utils import (
    fill_nans_in_arrays,
)


def test_fill_nans_in_arrays():
    """Test the fill_nans_in_arrays function"""
    array_with_nans = np.array([1.0, np.nan, 3.0, np.nan])
    nested_dict = {
        "array1": array_with_nans,
        "nested": {"array2": np.array([np.nan, 2.0, np.nan, 4.0])},
        "string_key": "not_an_array",
    }

    result = fill_nans_in_arrays(nested_dict)

    assert np.array_equal(result["array1"], np.array([1.0, 0.0, 3.0, 0.0]))
    assert np.array_equal(result["nested"]["array2"], np.array([0.0, 2.0, 0.0, 4.0]))
    assert result["string_key"] == "not_an_array"


def test_fill_nans_on_numpy_samples(config_filename):
    """Test the fill_nans_in_arrays function from configuration"""

    configuration = load_yaml_configuration(config_filename)
    # set custom satellite and nwp values, generation can be left as default 0.0
    configuration.input_data.satellite.dropout_value = -1.0
    configuration.input_data.nwp["ukv"].dropout_value = -2.0

    array_with_nans = np.array([1.0, np.nan, 3.0, np.nan])
    # we use copy() to ensure separate arrays for each key
    sample = {
        "generation": array_with_nans.copy(),
        "satellite_actual": array_with_nans.copy(),
        "ukv": {
            "nwp": np.array([np.nan, 2.0, np.nan, 4.0]),
        },
    }

    result = fill_nans_in_arrays(sample, config=configuration)

    assert np.array_equal(result["generation"], np.array([1.0, 0.0, 3.0, 0.0]))
    assert np.array_equal(result["satellite_actual"], np.array([1.0, -1.0, 3.0, -1.0]))
    assert np.array_equal(result["ukv"]["nwp"], np.array([-2.0, 2.0, -2.0, 4.0]))
