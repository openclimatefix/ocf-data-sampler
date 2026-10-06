"""Types for PVNet dataset orchestration."""

from typing import TypeAlias, TypedDict

import numpy as np
import torch

from ocf_data_sampler.common.types import TArray

SourceDict: TypeAlias = dict[str, TArray | dict[str, TArray]]
NumpySample: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
NumpyBatch: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
TensorBatch: TypeAlias = dict[str, torch.Tensor | dict[str, torch.Tensor]]


class SourceNormalisationArrays(TypedDict):
    """Broadcastable normalisation parameters for one source."""

    mean: np.ndarray
    std: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray


class NormalisationArrays(TypedDict, total=False):
    """Precomputed normalisation parameters by source."""

    nwp: dict[str, SourceNormalisationArrays]
    sat: SourceNormalisationArrays
