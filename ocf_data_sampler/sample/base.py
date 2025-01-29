"""
Base class definition - abstract
Handling of both flat and nested structures - consideration for NWP
"""

import logging
import numpy as np

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
