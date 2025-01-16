# base.py

"""
Base class definition - abstract
Handling of both flat and nested structures - consideration for NWP
"""

import numpy as np
import torch
import xarray as xr
import logging

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, ClassVar
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class SampleBase(ABC):
    """ 
    Abstract base class for all sample types 
    Provides core data storage and type validation
    """

    # Supported array types / format(s) definition
    VALID_ARRAY_TYPES = (np.ndarray, torch.Tensor, xr.DataArray)
    SUPPORTED_FORMATS: ClassVar[set] = {'.pt'}

    def __init__(self):
        """ Initialise data container """
        self._data: Dict[str, Any] = {}
    
    def __getitem__(self, key: str) -> Any:
        """ Retrieve item from sample """
        if key not in self._data:
            raise KeyError(f"Key {key} not in sample")
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """ Set item with type validation """
        self._validate_value(value)
        self._data[key] = value

    def keys(self) -> list[str]:
        """ Return sample keys """
        return list(self._data.keys())

    def _validate_value(self, value: Any) -> None:
        """ 
        Validate value is valid array type 
        Supports both single arrays and nested dicts i.e. NWP
        """
        if isinstance(value, dict):
            for k, v in value.items():
                self._validate_value(v)
        elif not isinstance(value, self.VALID_ARRAY_TYPES) and value is not None:
            raise TypeError(
                f"Value must be {self.VALID_ARRAY_TYPES} or nested dict "
                f"of these types - obtained {type(value)}"
            )

    @abstractmethod
    def plot(self, **kwargs) -> Optional[Any]:
        """ Abstract method for plotting """
        raise NotImplementedError("Plotting method must be implemented by subclass")

    @abstractmethod
    def validate(self) -> None:
        """ Abstract method for sample specific validation """
        raise NotImplementedError("Validation method must be implemented by subclass")

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """ Abstract method for saving sample data """
        raise NotImplementedError("Save method must be implemented by subclass")

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'SampleBase':
        """ Abstract class method for loading sample data """
        raise NotImplementedError("Load method must be implemented by subclass")
