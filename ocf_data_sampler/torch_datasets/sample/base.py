"""Base class for handling flat/nested data structures with NWP consideration."""

from abc import ABC, abstractmethod

import numpy as np
import torch

from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample, TensorBatch


class SampleBase(ABC):
    """Abstract base class for all sample types."""

    @abstractmethod
    def to_numpy(self) -> NumpySample:
        """Convert sample data to numpy format."""
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        """Create a visualisation of the data."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Saves the sample to disk in the implementations' required format."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "SampleBase":
        """Load a sample from disk from the implementations' format."""
        raise NotImplementedError


def batch_to_tensor(batch: NumpyBatch) -> TensorBatch:
    """Recursively converts numpy arrays in nested dict to torch tensors.

    Args:
        batch: NumpyBatch with data in numpy arrays
    Returns:
        TensorBatch with data in torch tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = batch_to_tensor(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                batch[k] = torch.tensor(v, dtype=torch.bool)
            elif np.issubdtype(v.dtype, np.number):
                batch[k] = torch.as_tensor(v)
    return batch


def copy_batch_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively copies tensors in nested dict to specified device.

    Args:
        batch: Nested dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to the new device
    """
    batch_copy = {}

    for k, v in batch.items():
        if isinstance(v, dict):
            batch_copy[k] = copy_batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch_copy[k] = v.to(device)
        else:
            batch_copy[k] = v
    return batch_copy
