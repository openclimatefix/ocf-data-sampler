"""Utility functions for merging dictionaries and filling NaNs in arrays."""

import numpy as np

from ocf_data_sampler.config.model import Configuration
from ocf_data_sampler.numpy_sample.gsp import GSPSampleKey
from ocf_data_sampler.numpy_sample.nwp import NWPSampleKey
from ocf_data_sampler.numpy_sample.satellite import SatelliteSampleKey
from ocf_data_sampler.numpy_sample.site import SiteSampleKey


def merge_dicts(list_of_dicts: list[dict]) -> dict:
    """Merge a list of dictionaries into a single dictionary."""
    # TODO: This doesn't account for duplicate keys, which will be overwritten
    combined_dict = {}
    for d in list_of_dicts:
        combined_dict.update(d)
    return combined_dict


def fill_nans_in_arrays(
    sample: dict, config: Configuration | None = None, nwp_provider: str | None = None,
) -> dict:
    """Fills all NaN values in each np.ndarray in the sample dictionary.

    Operation is performed in-place on the sample.
    By default a fill value of 0.0 are used, but if a config is provided,
    it can use the configured dropout values.
    """
    for k, v in sample.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if np.isnan(v).any():
                fill_value = 0.0
                if config is not None:
                    if k == GSPSampleKey.gsp:
                        fill_value = config.input_data.gsp.dropout_value
                    elif k == SiteSampleKey.generation:
                        fill_value = config.input_data.site.dropout_value
                    elif k == SatelliteSampleKey.satellite_actual:
                        fill_value = config.input_data.satellite.dropout_value
                    elif k == NWPSampleKey.nwp and nwp_provider in config.input_data.nwp:
                        fill_value = config.input_data.nwp[nwp_provider].dropout_value

                sample[k] = np.nan_to_num(v, copy=False, nan=fill_value)

        # Recursion is included to reach NWP arrays in subdict
        elif isinstance(v, dict):
            fill_nans_in_arrays(v, config=config, nwp_provider=k)

    return sample
