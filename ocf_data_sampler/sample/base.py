"""
Base class definition - abstract
Handling of both flat and nested structures - consideration for NWP
"""

import logging
import numpy as np
import torch
import xarray as xr

from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeAlias
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

NumpySample: TypeAlias = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
NumpyBatch: TypeAlias = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
TensorBatch: TypeAlias = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


class SampleBase(ABC):
    """ 
    Abstract base class for all sample types 
    Provides core data storage functionality
    """

    def __init__(self, data: Optional[Union[NumpySample, xr.Dataset]] = None):
        """ Initialise data container """
        logger.debug("Initialising SampleBase instance")
        self._data = data

    @abstractmethod
    def to_numpy(self) -> NumpySample:
        """ Convert data to a numpy array representation """
        raise NotImplementedError

    @abstractmethod
    def plot(self, **kwargs) -> None:
        """ Abstract method for plotting """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """ Abstract method for saving sample data """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'SampleBase':
        """ Abstract class method for loading sample data """
        raise NotImplementedError


def batch_to_tensor(batch: NumpyBatch) -> TensorBatch:
    """
    Moves ndarrays in a nested dict to torch tensors
    Args:
        batch: NumpyBatch with data in numpy arrays
    Returns:
        TensorBatch with data in torch tensors
    """
    if not batch:
        raise ValueError("Cannot convert empty batch to tensors")

    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = batch_to_tensor(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                batch[k] = torch.tensor(v, dtype=torch.bool)
            elif np.issubdtype(v.dtype, np.number):
                batch[k] = torch.as_tensor(v)
    return batch

import torch

def copy_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves tensor leaves in a nested dict to a new device.

    Args:
        batch: Nested dict with tensors to move.
        device: Device to move tensors to.

    Returns:
        A dict with tensors moved to the new device.
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
