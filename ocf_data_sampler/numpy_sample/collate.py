from ocf_data_sampler.numpy_sample import NWPSampleKey

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)



def stack_np_samples_into_batch(dict_list):
#     """
#     Stacks Numpy samples into a batch

#     Args:
#         dict_list: A list of dict-like Numpy samples to stack

#     Returns:
#         The stacked NumpySample object, aka a batch
#     """

    if not dict_list:
        raise ValueError("Input is empty")

    # Extract keys from first dict - structure
    sample = {}
    sample_keys = list(dict_list[0].keys())

    # Process - handle NWP separately due to nested structure
    for sample_key in sample_keys:
        if sample_key == "nwp":
            sample["nwp"] = process_nwp_data(dict_list)
        else:
            # Stack arrays for the given key across all dicts
            sample[sample_key] = stack_data_list([d[sample_key] for d in dict_list], sample_key)
    return sample


def process_nwp_data(dict_list):
    """Stacks data for NWP, handling nested structure"""
    
    nwp_sample = {}
    nwp_sources = dict_list[0]["nwp"].keys()

    # Stack data for each NWP source independently
    for nwp_source in nwp_sources:
        nested_keys = dict_list[0]["nwp"][nwp_source].keys()
        nwp_sample[nwp_source] = {
            key: stack_data_list([d["nwp"][nwp_source][key] for d in dict_list], key)
            for key in nested_keys
        }
    return nwp_sample

def _key_is_constant(sample_key):
    return sample_key.endswith("t0_idx") or sample_key == NWPSampleKey.channel_names


def stack_data_list(data_list: list,sample_key: Union[str, NWPSampleKey],):
    """How to combine data entries for each key

     Args:
        data_list: List of data entries to combine
        sample_key: Key identifying the data type
    """
    if _key_is_constant(sample_key):
        # These are always the same for all examples.
        return data_list[0]
    try:
        return np.stack(data_list)
    except Exception as e:
        logger.debug(f"Could not stack the following shapes together, ({sample_key})")
        shapes = [example.shape for example in data_list]
        logger.debug(shapes)
        logger.error(e)
        raise e
