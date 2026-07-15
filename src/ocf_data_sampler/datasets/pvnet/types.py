"""Types for PVNet dataset orchestration."""

from typing import TypeAlias

import numpy as np
import torch

from ocf_data_sampler.common.types import TArray

SourceDict: TypeAlias = dict[str, TArray | dict[str, TArray]]
NumpySample: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
NumpyBatch: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
TensorBatch: TypeAlias = dict[str, torch.Tensor | dict[str, torch.Tensor]]
