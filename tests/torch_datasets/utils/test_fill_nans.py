import numpy as np
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.torch_datasets.utils.fill_nans import fill_nans_in_dataset_dicts


def test_fill_nans_in_dataset_dicts(config_filename):
    """Test the fill_nans_in_arrays function from configuration"""

    configuration = load_yaml_configuration(config_filename)

    # Set custom satellite and nwp values, generation is left as default 0.0
    configuration.input_data.satellite.dropout_value = -1.0
    configuration.input_data.nwp["ukv"].dropout_value = -2.0

    gen = np.array([1.0, np.nan, 3.0, np.nan])
    sat = np.array([1.0, np.nan, 3.0, np.nan])
    ukv = np.array([np.nan, 3.0, np.nan])

    datasets_dict = {
        "generation": xr.DataArray(gen),
        "sat": xr.DataArray(sat),
        "nwp": {"ukv": xr.DataArray(ukv)},
    }

    datasets_dict = fill_nans_in_dataset_dicts(datasets_dict, config=configuration)

    assert np.array_equal(datasets_dict["generation"].values, np.array([1.0, 0.0, 3.0, 0.0]))
    assert np.array_equal(datasets_dict["sat"].values, np.array([1.0, -1.0, 3.0, -1.0]))
    assert np.array_equal(datasets_dict["nwp"]["ukv"].values, np.array([-2.0, 3.0, -2.0]))
