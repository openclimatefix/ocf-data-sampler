# base.py

"""
Base class definition - abstract
Handling of both flat and nested structures - consideration for NWP
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SampleBase(ABC):
    """ 
    Abstract base class for all sample types 
    Provides core data storage functionality
    """

    def __init__(self):
        """ Initialise data container """
        logger.debug("Initialising SampleBase instance")
        self._data: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        """ Basic dictionary-like access """
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """ Basic dictionary-like setting """
        self._data[key] = value

    def keys(self):
        """ Return available keys """
        return self._data.keys()

    @abstractmethod
    def validate(self) -> None:
        """ Abstract method for sample specific validation """
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
