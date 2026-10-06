"""Read, load, and convert PVNet source payloads to float32."""

import numpy as np
from xarray_tensorstore import read as xtr_read

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.datasets.pvnet.types import SourceDict

GENERATION_KEYS = frozenset({"generation", "generation_input", "generation_target"})



def initiate_reads(dataset_dict: SourceDict) -> SourceDict:
    """Initiate source reads in-place before waiting for any one source."""
    # Generation and its sliced input/target views are already loaded.
    # Kick off the tensorstore async reading
    for k, v in dataset_dict.items():
        if k in GENERATION_KEYS:
            continue
        if isinstance(v, dict):
            dataset_dict[k] = initiate_reads(v)
        else:
            if isinstance(v, LightDataArray):
                dataset_dict[k].read()
            else:
                dataset_dict[k] = xtr_read(v)
    return dataset_dict


def block_until_loaded(dataset_dict: SourceDict) -> SourceDict:
    """Block until source arrays are loaded into memory, modifying the dictionary in-place."""
    # Generation is eagerly loaded by open_generation, including its sliced input/target views.
    for k, v in dataset_dict.items():
        if k in GENERATION_KEYS:
            continue
        if isinstance(v, dict):
            dataset_dict[k] = block_until_loaded(v)
        else:
            dataset_dict[k] = v.load()
    return dataset_dict


def convert_to_float32(dataset_dict: SourceDict) -> SourceDict:
    """Convert loaded source payloads to float32 in-place, preserving coordinates."""
    for value in dataset_dict.values():
        if isinstance(value, dict):
            convert_to_float32(value)
        else:
            value.data = value.data.astype(np.float32, copy=False)
    return dataset_dict


def materialise_data(dataset_dict: SourceDict) -> SourceDict:
    """Read and load source arrays into memory, then convert their payloads to float32."""
    dataset_dict = initiate_reads(dataset_dict)
    dataset_dict = block_until_loaded(dataset_dict)
    return convert_to_float32(dataset_dict)
