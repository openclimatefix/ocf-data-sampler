""" 
PVNet - UK Regional sample / dataset implementation
"""

import numpy as np
import pandas as pd
import torch
import logging

from typing import Dict, Any, Union, List, Optional
from pathlib import Path

from ocf_data_sampler.numpy_sample import (
    NWPSampleKey, 
    GSPSampleKey,
    SatelliteSampleKey
)

from ocf_data_sampler.sample.base import SampleBase

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


logger = logging.getLogger(__name__)


class UKRegionalSample(SampleBase):
    """ Sample class specific to UK Regional PVNet """

    def __init__(self):
        logger.debug("Initialise UKRegionalSample instance")
        super().__init__()
        self._data = {}

    def to_numpy(self) -> Dict[str, Any]:
        """ Convert sample data to numpy format """
        logger.debug("Converting sample data to numpy format")
        return self._data

    def save(self, path: Union[str, Path]) -> None:
        """ Save PVNet sample as .pt """
        logger.debug(f"Saving UKRegionalSample to {path}")
        path = Path(path)
        
        if path.suffix != '.pt':
            logger.error(f"Invalid file format: {path.suffix}")
            raise ValueError(f"Only .pt format is supported: {path.suffix}")
        
        torch.save(self._data, path)
        logger.debug(f"Successfully saved UKRegionalSample to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UKRegionalSample':
        """ Load PVNet sample data from .pt """
        logger.debug(f"Attempting to load UKRegionalSample from {path}")
        path = Path(path)
        
        if path.suffix != '.pt':
            logger.error(f"Invalid file format: {path.suffix}")
            raise ValueError(f"Only .pt format is supported: {path.suffix}")
        
        instance = cls()
        # TODO: We should move away from using torch.load(..., weights_only=False)
        # This is not recommended
        instance._data = torch.load(path, weights_only=False)
        logger.debug(f"Successfully loaded UKRegionalSample from {path}")
        return instance

    def plot(self, **kwargs) -> None:
        """ Sample visualisation definition """
        logger.debug("Creating UKRegionalSample visualisation")

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Matplotlib required for plotting"
                "Install via 'ocf_data_sampler[plot]'"
            )

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
            
            if GSPSampleKey.solar_azimuth in self._data and GSPSampleKey.solar_elevation in self._data:
                logger.debug("Plotting solar position data")
                axes[1, 1].plot(self._data[GSPSampleKey.solar_azimuth], label='Azimuth')
                axes[1, 1].plot(self._data[GSPSampleKey.solar_elevation], label='Elevation')
                axes[1, 1].set_title('Solar Position')
                axes[1, 1].legend()

            if SatelliteSampleKey.satellite_actual in self._data:
                logger.debug("Plotting satellite data")
                axes[1, 0].imshow(self._data[SatelliteSampleKey.satellite_actual])
                axes[1, 0].set_title('Satellite Data')
            
            plt.tight_layout()
            plt.show()
            logger.debug("Successfully created visualisation")
        except Exception as e:
            logger.error(f"Error creating visualisation: {str(e)}")
            raise
