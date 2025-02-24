"""Utility functions for merging dictionaries and filling NaNs in arrays."""

import numpy as np


def merge_dicts(list_of_dicts: list[dict]) -> dict:
    """Merge a list of dictionaries into a single dictionary."""
    # TODO: This doesn't account for duplicate keys, which will be overwritten
    combined_dict = {}
    for d in list_of_dicts:
        combined_dict.update(d)
    return combined_dict


def fill_nans_in_arrays(sample: dict) -> dict:
    """Fills all NaN values in each np.ndarray in the sample dictionary with zeros.

    Operation is performed in-place on the sample.
    """
    for k, v in sample.items():
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            if np.isnan(v).any():
                sample[k] = np.nan_to_num(v, copy=False, nan=0.0)

        # Recursion is included to reach NWP arrays in subdict
        elif isinstance(v, dict):
            fill_nans_in_arrays(v)

    return sample
