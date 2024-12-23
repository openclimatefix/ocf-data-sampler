from ocf_data_sampler.numpy_sample import NWPSampleKey

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


def stack_np_examples_into_sample(dict_list):
    """
    Stacks Numpy examples into a sample

    See also: `unstack_np_sample_into_examples()` for opposite

    Args:
        dict_list: A list of dict-like Numpy examples to stack

    Returns:
        The stacked NumpySample object
    """
    sample = {}

    sample_keys = list(dict_list[0].keys())

    for sample_key in sample_keys:
        # NWP is nested so treat separately
        if sample_key == "nwp":
            nwp_sample: dict[str, NWPSampleKey] = {}

            # Unpack source keys
            nwp_sources = list(dict_list[0]["nwp"].keys())

            for nwp_source in nwp_sources:
                # Keys can be different for different NWPs
                nwp_sample_keys = list(dict_list[0]["nwp"][nwp_source].keys())

                nwp_source_sample = {}
                for nwp_sample_key in nwp_sample_keys:
                    nwp_source_sample[nwp_sample_key] = stack_data_list(
                        [d["nwp"][nwp_source][nwp_sample_key] for d in dict_list],
                        nwp_sample_key,
                    )

                nwp_sample[nwp_source] = nwp_source_sample

            sample["nwp"] = nwp_sample

        else:
            sample[sample_key] = stack_data_list(
                [d[sample_key] for d in dict_list],
                sample_key,
            )

    return sample


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
