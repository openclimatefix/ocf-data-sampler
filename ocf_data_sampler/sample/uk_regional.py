# uk_regional.py

""" 
PVNet - UK Regional sample / dataset implementation
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging

from typing import Dict, Any, Union, List
from pathlib import Path

from ocf_data_sampler.numpy_sample import (
    NWPSampleKey, 
    GSPSampleKey,
    SatelliteSampleKey
)

from ocf_data_sampler.sample.base import SampleBase


logger = logging.getLogger(__name__)


class PVNetSample(SampleBase):
    """ Sample class specific to PVNet """
    
    # Feature space definitions - core
    CORE_KEYS = {
        NWPSampleKey.nwp
    }

    # Feature space definitions - optional
    OPTIONAL_KEYS = {
        GSPSampleKey.gsp,
        GSPSampleKey.solar_azimuth,
        GSPSampleKey.solar_elevation,
        SatelliteSampleKey.satellite_actual
    }

    def __init__(self):
        logger.debug("Initialise PVNetSample instance")
        super().__init__()

    def validate(self) -> None:
        logger.debug("Validating PVNetSample")
        
        # Check required keys - feature space validation
        missing_keys = self.CORE_KEYS - set(self.keys())
        if missing_keys:
            logger.error(f"Validation failed: {missing_keys}")
            raise ValueError(f"Missing required keys: {missing_keys}")

        # Validate NWP structure
        if NWPSampleKey.nwp in self._data:
            if not isinstance(self._data[NWPSampleKey.nwp], dict):
                logger.error("Validation failed")
                raise TypeError("NWP data must be nested dictionary")
            
            # Additional NWP-specific validations
            nwp_sources = self._data[NWPSampleKey.nwp].keys()
            for source in nwp_sources:
                if not all(key in self._data[NWPSampleKey.nwp][source] 
                           for key in ['nwp', NWPSampleKey.channel_names]):
                    logger.error(f"Incomplete NWP data for source {source}")
                    raise ValueError(f"Missing keys in NWP source {source}")
        
        # Validate timestep consistency
        gsp_timesteps = len(self._data[GSPSampleKey.gsp])
        time_dependent_keys = [
            GSPSampleKey.solar_azimuth,
            GSPSampleKey.solar_elevation
        ]
        
        # Add satellite to validation if present
        if SatelliteSampleKey.satellite_actual in self._data:
            time_dependent_keys.append(SatelliteSampleKey.satellite_actual)
        
        for key in time_dependent_keys:
            if len(self._data[key]) != gsp_timesteps:
                logger.error(f"Validation failed - inconsistent timesteps for {key}")
                raise ValueError("Inconsistent number of timesteps")
                
        logger.debug("PVNetSample validation successful")

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """ Convert sample to numpy arrays - nested handling """
        logger.debug("Converting sample to numpy format")
        
        def convert_to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.numpy()
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, dict):
                return {k: convert_to_numpy(v) for k, v in data.items()}
            else:
                return data

        try:
            numpy_data = {k: convert_to_numpy(v) for k, v in self._data.items()}
            logger.debug("Successfully converted to numpy format")
            return numpy_data
            
        except Exception as e:
            logger.error(f"Error converting to numpy: {str(e)}")
            raise

    def save(self, path: Union[str, Path]) -> None:
        """ Save PVNet sample as .pt """
        logger.debug(f"Saving PVNetSample to {path}")
        path = Path(path)
        
        if path.suffix != '.pt':
            logger.error(f"Invalid file format: {path.suffix}")
            raise ValueError(f"Only .pt format is supported: {path.suffix}")
        
        try:
            self.validate()            
            torch.save(self._data, path)
            logger.debug(f"Successfully saved PVNetSample to {path}")
        except Exception as e:
            logger.error(f"Error saving to {path}: {str(e)}")
            raise

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PVNetSample':
        """ Load PVNet sample data from .pt """
        logger.debug(f"Attempting to load PVNetSample from {path}")
        path = Path(path)
        
        if path.suffix != '.pt':
            logger.error(f"Invalid file format: {path.suffix}")
            raise ValueError(f"Only .pt format is supported: {path.suffix}")
        
        try:
            instance = cls()
            instance._data = torch.load(path)
            instance.validate()
            logger.debug(f"Successfully loaded PVNetSample from {path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading from {path}: {str(e)}")
            raise

    def plot(self, **kwargs) -> None:
        """ Sample visualisation definition """
        logger.debug("Creating PVNetSample visualisation")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            if NWPSampleKey.nwp in self._data:
                logger.debug("Plotting NWP data")
                first_nwp = list(self._data[NWPSampleKey.nwp].values())[0]
                if 'nwp' in first_nwp:
                    axes[0, 1].imshow(first_nwp['nwp'][0])
                    axes[0, 1].set_title('NWP (First Channel)')
                    if NWPSampleKey.channel_names in first_nwp:
                        channel_names = first_nwp[NWPSampleKey.channel_names]
                        if len(channel_names) > 0:
                            axes[0, 1].set_title(f'NWP: {channel_names[0]}')

            if GSPSampleKey.gsp in self._data:
                logger.debug("Plotting GSP generation data")
                axes[0, 0].plot(self._data[GSPSampleKey.gsp])
                axes[0, 0].set_title('GSP Generation')
            
            if SatelliteSampleKey.satellite_actual in self._data:
                logger.debug("Plotting satellite data")
                axes[1, 0].imshow(self._data[SatelliteSampleKey.satellite_actual])
                axes[1, 0].set_title('Satellite Data')
            
            if GSPSampleKey.solar_azimuth in self._data and GSPSampleKey.solar_elevation in self._data:
                logger.debug("Plotting solar position data")
                axes[1, 1].plot(self._data[GSPSampleKey.solar_azimuth], label='Azimuth')
                axes[1, 1].plot(self._data[GSPSampleKey.solar_elevation], label='Elevation')
                axes[1, 1].set_title('Solar Position')
                axes[1, 1].legend()
            
            plt.tight_layout()
            plt.show()
            logger.debug("Successfully created visualisation")
        except Exception as e:
            logger.error(f"Error creating visualisation: {str(e)}")
            raise
