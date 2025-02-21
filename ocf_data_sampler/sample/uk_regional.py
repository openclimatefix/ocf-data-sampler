"""PVNet UK Regional sample implementation for dataset handling and visualisation"""

from typing_extensions import override

import torch
from matplotlib import pyplot as plt

from ocf_data_sampler.sample.base import SampleBase, NumpySample
from ocf_data_sampler.numpy_sample import (
    NWPSampleKey, 
    GSPSampleKey,
    SatelliteSampleKey
)


class UKRegionalSample(SampleBase):
    """Handles UK Regional PVNet data operations"""

    def __init__(self, data: NumpySample):
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:
        return self._data

    def save(self, path: str) -> None:
        """Save PVNet sample as pickle format using torch.save
        
        Args:
            path: Path to save the sample data to
        """        
        torch.save(self._data, path)

    @classmethod
    def load(cls, path: str) -> 'UKRegionalSample':
        """Load PVNet sample data from .pt format
        
        Args:
            path: Path to load the sample data from
        """
        # TODO: We should move away from using torch.load(..., weights_only=False)
        return cls(torch.load(path, weights_only=False))

    def plot(self) -> None:
        """Creates visualisations for NWP, GSP, solar position, and satellite data"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if NWPSampleKey.nwp in self._data:
            first_nwp = list(self._data[NWPSampleKey.nwp].values())[0]
            if 'nwp' in first_nwp:
                axes[0, 1].imshow(first_nwp['nwp'][0])
                title = 'NWP (First Channel)'
                if NWPSampleKey.channel_names in first_nwp:
                    channel_names = first_nwp[NWPSampleKey.channel_names]
                    if channel_names:
                        title = f'NWP: {channel_names[0]}'
                axes[0, 1].set_title(title)

        if GSPSampleKey.gsp in self._data:
            axes[0, 0].plot(self._data[GSPSampleKey.gsp])
            axes[0, 0].set_title('GSP Generation')
        
        if GSPSampleKey.solar_azimuth in self._data and GSPSampleKey.solar_elevation in self._data:
            axes[1, 1].plot(self._data[GSPSampleKey.solar_azimuth], label='Azimuth')
            axes[1, 1].plot(self._data[GSPSampleKey.solar_elevation], label='Elevation')
            axes[1, 1].set_title('Solar Position')
            axes[1, 1].legend()

        if SatelliteSampleKey.satellite_actual in self._data:
            axes[1, 0].imshow(self._data[SatelliteSampleKey.satellite_actual])
            axes[1, 0].set_title('Satellite Data')
        
        plt.tight_layout()
        plt.show()
