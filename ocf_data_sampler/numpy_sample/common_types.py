"""This module defines type aliases for numpy and torch data structures used in the project."""


from typing import TypeAlias

import numpy as np
import torch

NumpySample: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
NumpyBatch: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
TensorBatch: TypeAlias = dict[str, torch.Tensor | dict[str, torch.Tensor]]
