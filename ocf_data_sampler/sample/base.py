"""
Base class definition - abstract
Handling of both flat and nested structures - consideration for NWP
"""

import logging
import numpy as np
import torch

from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeAlias
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

NumpySample: TypeAlias = Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]
TensorSample: TypeAlias = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


class SampleBase(ABC):
    """ 
    Abstract base class for all sample types 
    Provides core data storage functionality
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """ Initialise data container """
        logger.debug("Initialising SampleBase instance")
        self._data = data if data is not None else {}

    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    @abstractmethod
    def to_numpy(self) -> Dict[str, Any]:
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


class NumpySample(SampleBase):
    """
    Sample implementation 
    Numpy arrays - nested support
    """

    def __getitem__(self, key: str) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Union[np.ndarray, Dict[str, np.ndarray]]) -> None:
        self._data[key] = value

    def to_numpy(self) -> Dict[str, Any]:
        return self._data
    
    def plot(self, **kwargs) -> None:
        raise NotImplementedError

    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'NumpySample':
        raise NotImplementedError


class TensorSample(SampleBase):
    """
    Sample implementation
    PyTorch tensors - nested support
    """

    def __getitem__(self, key: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._data[key]
    
    def __setitem__(self, key: str, value: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
        self._data[key] = value

    def to_numpy(self) -> Dict[str, Any]:
        numpy_data = {}
        for key, value in self._data.items():
            if isinstance(value, dict):
                numpy_data[key] = {k: v.cpu().numpy() for k, v in value.items()}
            else:
                numpy_data[key] = value.cpu().numpy()
        return numpy_data
    
    def plot(self, **kwargs) -> None:
        raise NotImplementedError

    def save(self, path: Union[str, Path]) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TensorSample':
        raise NotImplementedError


def sample_to_tensor(sample: NumpySample) -> TensorSample:
    """
    Moves data in a NumpySample to a TensorSample

    Args:
        sample: NumpySample with data in numpy arrays

    Returns:
        TensorSample with data in torch tensors
    """
    return _sample_to_tensor(sample)
    

def _sample_to_tensor(sample: dict) -> dict:
    """
    Moves arrays in a nested dict to torch tensors

    Args:
        sample: nested dict with data in numpy arrays

    Returns:
        Nested dict with data in torch tensors
    """
    for k, v in sample.items():
        if isinstance(v, dict):
            sample[k] = _sample_to_tensor(v)
        elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            sample[k] = torch.as_tensor(v)
    return sample
