# base.py

"""
Base class definition
Handling of both flat and nested structures - consideration for NWP
"""

import numpy as np
import torch
import xarray as xr
import logging

from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class SampleBase(ABC):
    """ Base class for all considered sample types """

    # Key constant definitions
    # Specific array types and formats
    VALID_ARRAY_TYPES = (np.ndarray, torch.Tensor, xr.DataArray)
    SUPPORTED_FORMATS = {'.nc', '.zarr', '.npz'}

    # Container
    # Dict for flat and nested arrays
    def __init__(self):
        """ Initialisation """
        self._data: Dict[str, Any] = {}
    
    # Following two functions set dict with validation
    def __getitem__(self, key: str) -> Any:
        """ Get item """
        if key not in self._data:
            raise KeyError(f"Key {key} not in sample")
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """ Set item """
        self._validate_value(value)
        self._data[key] = value

    def keys(self) -> list[str]:
        """ Get keys """
        return list(self._data.keys())    

    # Main I / O function operation
    def save(self, path: Union[str, Path]) -> None:
        """ Save to disk standard function """
        path = Path(path)
        if path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {path.suffix}"
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )
        
        save_dict = self.prepare_for_save()
        
        # Format handling stage
        if path.suffix == '.nc':
            ds = self._dict_to_dataset(save_dict)
            ds.to_netcdf(path)
            ds.close()
        elif path.suffix == '.zarr':
            ds = self._dict_to_dataset(save_dict)
            ds.to_zarr(path, mode='w')
            ds.close()
        else:

            # Handle nested structures for npz
            np_data = {}
            for key, value in save_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        np_data[f"{key}/{sub_key}"] = sub_value
                else:
                    np_data[key] = value
            np.savez_compressed(path, **np_data)

    def _validate_value(self, value: Any) -> None:
        """ Validation - value is valid array or nested dict of valid arrays """
        if isinstance(value, dict):
            for k, v in value.items():
                self._validate_value(v)
        elif not isinstance(value, self.VALID_ARRAY_TYPES) and value is not None:
            raise TypeError(
                f"Value must be {self.VALID_ARRAY_TYPES} or nested dict"
                f"of these types - obtained {type(value)}"
                )

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SampleBase':
        """ Load sample data """
        path = Path(path)
        if path.suffix not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format {path.suffix}"
                f"Supported formats: {cls.SUPPORTED_FORMATS}"
            )
        
        instance = cls()

        # Format handling stage
        try:
            if path.suffix == '.nc':
                ds = xr.open_dataset(path)
                instance._data = cls._dataset_to_dict(ds)
                ds.close()
            elif path.suffix == '.zarr':
                ds = xr.open_zarr(path)
                instance._data = cls._dataset_to_dict(ds)
                ds.close()
            else:
                with np.load(path, allow_pickle=True) as data:
                    loaded_data = {}
                    for key in data.files:
                        # Parse key - handle nested
                        if '/' in key:
                            main_key, sub_key = key.split('/')
                            # Init nested dict
                            if main_key not in loaded_data:
                                loaded_data[main_key] = {}
                            loaded_data[main_key][sub_key] = data[key]
                        else:
                            loaded_data[key] = data[key]
                    instance._data = loaded_data
        except Exception as e:
            logger.error(f"Error loading {path}: {str(e)}")
            raise
                
        return instance

    # Type conversion operations
    def to_torch(self) -> 'SampleBase':
        def numpy_to_torch(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, xr.DataArray):
                return torch.from_numpy(x.values)
            else:
                return torch.from_numpy(x)
        
        self._convert_arrays(numpy_to_torch)
        return self

    def to_numpy(self) -> 'SampleBase':
        self._convert_arrays(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x)
        return self
    
    @abstractmethod
    def plot(self, **kwargs) -> None:
        pass

    def fill_nans(self, fill_value: float = 0.0) -> None:
        """ Fill NaN values """
        def fill_array_nans(arr: Union[np.ndarray, torch.Tensor, xr.DataArray]) -> Union[np.ndarray, torch.Tensor]:
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
                return np.nan_to_num(arr, nan=fill_value)
            elif isinstance(arr, torch.Tensor):
                return torch.nan_to_num(arr, nan=fill_value)
            elif isinstance(arr, xr.DataArray):
                return arr.fillna(fill_value).values
            return arr
            
        for key, value in self._data.items():
            if isinstance(value, self.VALID_ARRAY_TYPES):
                self._data[key] = fill_array_nans(value)
            elif isinstance(value, dict):
                self._data[key] = {
                    k: fill_array_nans(v) if isinstance(v, self.VALID_ARRAY_TYPES) else v
                    for k, v in value.items()
                }
    
    def _numpy_to_torch(self, arr: Union[np.ndarray, xr.DataArray]) -> torch.Tensor:
        if isinstance(arr, xr.DataArray):
            logger.debug("Converting to torch tensor")
            return torch.from_numpy(arr.values)
        elif isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        else:
            raise TypeError(f"Cannot convert {type(arr)} to tensor")
    
    def _torch_to_numpy(self, tensor: Union[torch.Tensor, np.ndarray, xr.DataArray]) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        elif isinstance(tensor, xr.DataArray):
            logger.debug("Converting to numpy array")
            return tensor.values
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise TypeError(f"Cannot convert {type(tensor)} to numpy array")
    
    # Major util function for array conversion
    def _convert_arrays(self, convert_fn: Callable[[Any], Any]) -> None:
        """ Convert arrays in sample - for both flat and nested dicts """
        for key, value in self._data.items():
            if isinstance(value, self.VALID_ARRAY_TYPES):
                try:
                    self._data[key] = convert_fn(value)
                except Exception as e:
                    logger.error(f"Error converting array at {key}: {e}")
                    raise
            elif isinstance(value, dict):
                self._data[key] = {
                    k: convert_fn(v) if isinstance(v, self.VALID_ARRAY_TYPES) else v
                    for k, v in value.items()
                }

    # Format conversion
    # Preserves nested structure
    def _dict_to_dataset(self, data_dict: Dict[str, Any]) -> xr.Dataset:

        arrays = {}
        coords = {}
        
        for key, value in data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                        var_name = f"{key}_{sub_key}"
                        dims = [f"{var_name}_dim_{i}" for i in range(sub_value.ndim)]
                        arrays[var_name] = (dims, self._to_numpy(sub_value))
            else:
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    dims = [f"{key}_dim_{i}" for i in range(value.ndim)]
                    arrays[key] = (dims, self._to_numpy(value))
        
        return xr.Dataset(arrays, coords=coords)

    # Reconstruction of nested structure
    @staticmethod
    def _dataset_to_dict(ds: xr.Dataset) -> Dict[str, Any]:

        data_dict = {}
        for var_name in ds.variables:
            if var_name in ds.coords:
                continue
                
            if '_' in var_name:
                main_key, sub_key = var_name.split('_', 1)
                if main_key not in data_dict:
                    data_dict[main_key] = {}
                data_dict[main_key][sub_key] = ds[var_name].values
            else:
                data_dict[var_name] = ds[var_name].values
        return data_dict
    
    def _to_numpy(self, arr: Union[np.ndarray, torch.Tensor, xr.DataArray]) -> np.ndarray:
        if isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        elif isinstance(arr, xr.DataArray):
            return arr.values
        return arr
    
    def prepare_for_save(self) -> Dict[str, Any]:
        self.to_numpy()
        return self._data
