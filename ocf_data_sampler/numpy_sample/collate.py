"""Functions for collating samples into batches."""

import numpy as np

from ocf_data_sampler.numpy_sample.common_types import NumpyBatch


def stack_np_samples_into_batch(dict_list: list[dict]) -> NumpyBatch:
    """Stacks list of dict samples into a dict where all samples are joined along a new axis.

    Args:
        dict_list: A list of dict-like samples to stack

    Returns:
        Dict of the samples stacked with new batch dimension on axis 0
    """
    batch = {}

    keys = list(dict_list[0].keys())

    for key in keys:
        # NWP is nested so treat separately
        if key == "nwp":
            batch["nwp"] = {}

            # Unpack NWP provider keys
            nwp_providers = list(dict_list[0]["nwp"].keys())

            for nwp_provider in nwp_providers:
                # Keys can be different for different NWPs
                nwp_keys = list(dict_list[0]["nwp"][nwp_provider].keys())

                # Create dict to store NWP batch for this provider
                nwp_provider_batch = {}

                for nwp_key in nwp_keys:
                    # Stack values under each NWP key for this provider
                    nwp_provider_batch[nwp_key] = stack_data_list(
                        [d["nwp"][nwp_provider][nwp_key] for d in dict_list],
                        nwp_key,
                    )

                batch["nwp"][nwp_provider] = nwp_provider_batch

        else:
            batch[key] = stack_data_list([d[key] for d in dict_list], key)

    return batch


def _key_is_constant(key: str) -> bool:
    """Check if a key is for value which is constant for all samples."""
    return key.endswith("t0_idx") or key.endswith("channel_names")


def stack_data_list(data_list: list, key: str) -> np.ndarray:
    """Stack a sequence of data elements along a new axis.

    Args:
       data_list: List of data elements to combine
       key: string identifying the data type
    """
    if _key_is_constant(key):
        return data_list[0]
    else:
        return np.stack(data_list)
