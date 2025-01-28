# site.py

""" 
PVNet - Site sample / dataset implementation
"""

import logging
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

from ocf_data_sampler.sample.base import SampleBase

from ocf_data_sampler.torch_datasets.datasets.site import convert_netcdf_to_numpy_sample


logger = logging.getLogger(__name__)


class SiteSample(SampleBase):
    """ Sample class specific to Site PVNet """

    def __init__(self):
        logger.debug("Initialise SiteSample instance")
        super().__init__()

    def to_numpy(self) -> Dict[str, Any]:
        """ Convert sample numpy arrays - netCDF conversion """
        logger.debug("Converting site sample to numpy format")
        
        try:
            if not isinstance(self._data, xr.Dataset):
                raise TypeError("Data must be xarray Dataset")
            
            numpy_data = convert_netcdf_to_numpy_sample(self._data)
            
            # Work around addition - to be checked!
            # Wrap site data in dict
            if 'site' in numpy_data and not isinstance(numpy_data['site'], dict):
                numpy_data['site'] = {'site': numpy_data['site']}

            logger.debug("Successfully converted to numpy format")
            return numpy_data
            
        except Exception as e:
            logger.error(f"Error converting to numpy: {str(e)}")
            raise

    def save(self, path: Union[str, Path]) -> None:
        """ Save site sample as netCDF - h5netcdf engine """
        logger.debug(f"Saving SiteSample to {path}")
        path = Path(path)
        
        if path.suffix != '.nc':
            logger.error(f"Invalid file format - {path.suffix}")
            raise ValueError("Only .nc format is supported")
        
        try:
            if not isinstance(self._data, xr.Dataset):
                raise TypeError("Data must be xarray Dataset for saving")
                
            self._data.to_netcdf(
                path, 
                mode="w", 
                engine="h5netcdf"
            )
            logger.debug(f"Successfully saved SiteSample - {path}")
            
        except Exception as e:
            logger.error(f"Error saving - {path}: {str(e)}")
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SiteSample':
        """ Load site sample from netCDF """
        logger.debug(f"Loading SiteSample from {path}")
        path = Path(path)
        
        if path.suffix != '.nc':
            logger.error(f"Invalid file format - {path.suffix}")
            raise ValueError("Only .nc format is supported")
        
        try:
            instance = cls()
            instance._data = xr.open_dataset(path)
            logger.debug(f"Successfully loaded SiteSample - {path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading - {path}: {str(e)}")
            raise

    # TO DO - placeholder for now
    def plot(self, **kwargs) -> None:
        """ Plot sample data - placeholder """
        pass
