from ocf_data_sampler.numpy_sample import NWPSampleKey

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)



def stack_np_examples_into_sample(dict_list):
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
            sample[sample_key] = stack_data_for_key(dict_list, sample_key)
    return sample


def process_nwp_data(dict_list):
    nwp_sample = {}
    nwp_sources = dict_list[0]["nwp"].keys()

   # Stack data for each NWP source independently
    for nwp_source in nwp_sources:
        nwp_sample[nwp_source] = stack_nested_keys(dict_list, "nwp", nwp_source)
    return nwp_sample
def stack_data_for_key(dict_list, sample_key):
    # Stack arrays for given key across all dicts
    return stack_data_list([d[sample_key] for d in dict_list], sample_key)


def stack_nested_keys(dict_list, parent_key, child_key):
    # Processes two-level nested data 
    # Preserves hierarchy whilst stacking arrays
    nested_keys = dict_list[0][parent_key][child_key].keys()
    return {
        key: stack_data_list(
            [d[parent_key][child_key][key] for d in dict_list],
            key,
        )
        for key in nested_keys
    }


# def stack_np_examples_into_sample(dict_list):
#     """
#     Stacks Numpy examples into a sample

#     See also: `unstack_np_sample_into_examples()` for opposite

#     Args:
#         dict_list: A list of dict-like Numpy examples to stack

#     Returns:
#         The stacked NumpySample object
#     """
#     sample = {}

#     sample_keys = list(dict_list[0].keys())

#     for sample_key in sample_keys:
#         # NWP is nested so treat separately
#         if sample_key == "nwp":
#             nwp_sample: dict[str, NWPSampleKey] = {}

#             # Unpack source keys
#             nwp_sources = list(dict_list[0]["nwp"].keys())

#             for nwp_source in nwp_sources:
#                 # Keys can be different for different NWPs
#                 nwp_sample_keys = list(dict_list[0]["nwp"][nwp_source].keys())

#                 nwp_source_sample = {}
#                 for nwp_sample_key in nwp_sample_keys:
#                     nwp_source_sample[nwp_sample_key] = stack_data_list(
#                         [d["nwp"][nwp_source][nwp_sample_key] for d in dict_list],
#                         nwp_sample_key,
#                     )

#                 nwp_sample[nwp_source] = nwp_source_sample

#             sample["nwp"] = nwp_sample

#         else:
#             sample[sample_key] = stack_data_list(
#                 [d[sample_key] for d in dict_list],
#                 sample_key,
#             )

#     return sample


def _key_is_constant(sample_key):
    is_constant = sample_key.endswith("t0_idx") or sample_key == NWPSampleKey.channel_names
    return is_constant


def stack_data_list(
    data_list: list,
    sample_key: Union[str, NWPSampleKey],
):
    """How to combine data entries for each key
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
