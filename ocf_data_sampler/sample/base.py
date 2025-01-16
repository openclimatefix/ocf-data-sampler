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
from typing import Any, Dict, Optional, Union, ClassVar
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
        logger.debug("Initialising SampleBase instance")
        self._data: Dict[str, Any] = {}
    
    def __getitem__(self, key: str) -> Any:
        """ Retrieve item from sample """
        if key not in self._data:
            logger.error(f"Failed to retrieve key '{key}'")
            raise KeyError(f"Key {key} not in sample")
        logger.debug(f"Retrieved value for key '{key}'")
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """ Set item with type validation """
        logger.debug(f"Setting value for key '{key}'")
        try:
            self._validate_value(value)
            self._data[key] = value
            logger.debug(f"Successfully set value for key '{key}'")
        except TypeError as e:
            logger.error(f"Type validation failed for key '{key}': {str(e)}")
            raise

    def keys(self) -> list[str]:
        """ Return sample keys """
        logger.debug("Returning all sample keys")
        return list(self._data.keys())

    def _validate_value(self, value: Any) -> None:
        """ 
        Validate value is valid array type 
        Supports both single arrays and nested dicts i.e. NWP
        """
        logger.debug(f"Validating value of type {type(value)}")
        if isinstance(value, dict):
            logger.debug("Validating nested dictionary")
            for k, v in value.items():
                self._validate_value(v)
        elif not isinstance(value, self.VALID_ARRAY_TYPES) and value is not None:
            logger.error(f"Invalid value type: {type(value)}")
            raise TypeError(
                f"Value must be {self.VALID_ARRAY_TYPES} or nested dict "
                f"of these types - obtained {type(value)}"
            )
        logger.debug("Value validation successful")

    @abstractmethod
    def plot(self, **kwargs) -> Optional[Any]:
        """ Abstract method for plotting """
        logger.debug("Plot method called")
        raise NotImplementedError("Plotting method must be implemented by subclass")

    @abstractmethod
    def validate(self) -> None:
        """ Abstract method for sample specific validation """
        logger.debug("Validate method called")
        raise NotImplementedError("Validation method must be implemented by subclass")

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """ Abstract method for saving sample data """
        logger.debug(f"Save method called with path: {path}")
        raise NotImplementedError("Save method must be implemented by subclass")

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'SampleBase':
        """ Abstract class method for loading sample data """
        logger.debug(f"Load method called with path: {path}")
        raise NotImplementedError("Load method must be implemented by subclass")
