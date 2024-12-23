from ocf_data_sampler.numpy_batch import NWPBatchKey

import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


def stack_np_examples_into_batch(dict_list):
    """
    Stacks Numpy examples into a batch

    See also: `unstack_np_batch_into_examples()` for opposite

    Args:
        dict_list: A list of dict-like Numpy examples to stack

    Returns:
        The stacked NumpyBatch object
    """
    batch = {}

    batch_keys = list(dict_list[0].keys())

    for batch_key in batch_keys:
        # NWP is nested so treat separately
        if batch_key == "nwp":
            nwp_batch: dict[str, NWPBatchKey] = {}

            # Unpack source keys
            nwp_sources = list(dict_list[0]["nwp"].keys())

            for nwp_source in nwp_sources:
                # Keys can be different for different NWPs
                nwp_batch_keys = list(dict_list[0]["nwp"][nwp_source].keys())

                nwp_source_batch = {}
                for nwp_batch_key in nwp_batch_keys:
                    nwp_source_batch[nwp_batch_key] = stack_data_list(
                        [d["nwp"][nwp_source][nwp_batch_key] for d in dict_list],
                        nwp_batch_key,
                    )

                nwp_batch[nwp_source] = nwp_source_batch

            batch["nwp"] = nwp_batch

        else:
            batch[batch_key] = stack_data_list(
                [d[batch_key] for d in dict_list],
                batch_key,
            )

    return batch


def _key_is_constant(batch_key):
    is_constant = batch_key.endswith("t0_idx") or batch_key == NWPBatchKey.channel_names
    return is_constant


def stack_data_list(
    data_list: list,
    batch_key: Union[str, NWPBatchKey],
):
    """How to combine data entries for each key
    """
    if _key_is_constant(batch_key):
        # These are always the same for all examples.
        return data_list[0]
    try:
        return np.stack(data_list)
    except Exception as e:
        logger.debug(f"Could not stack the following shapes together, ({batch_key})")
        shapes = [example.shape for example in data_list]
        logger.debug(shapes)
        logger.error(e)
        raise e
