"""PVNet UK Regional sample implementation for dataset handling and visualisation."""

import torch
from typing_extensions import override

from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
)
from ocf_data_sampler.sample.base import NumpySample, SampleBase


class UKRegionalSample(SampleBase):
    """Handles UK Regional PVNet data operations."""

    def __init__(self, data: NumpySample) -> None:
        """Initialises UK Regional sample with data."""
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:
        return self._data

    @override
    def save(self, path: str) -> None:
        # Saves to pickle format
        torch.save(self._data, path)

    @classmethod
    @override
    def load(cls, path: str) -> "UKRegionalSample":
        # Loads from .pt format
        # TODO: We should move away from using torch.load(..., weights_only=False)
        return cls(torch.load(path, weights_only=False))

    @override
    def plot(self) -> None:
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        if NWPSampleKey.nwp in self._data:
            first_nwp = next(iter(self._data[NWPSampleKey.nwp].values()))
            if "nwp" in first_nwp:
                axes[0, 1].imshow(first_nwp["nwp"][0])
                title = "NWP (First Channel)"
                if NWPSampleKey.channel_names in first_nwp:
                    channel_names = first_nwp[NWPSampleKey.channel_names]
                    if channel_names:
                        title = f"NWP: {channel_names[0]}"
                axes[0, 1].set_title(title)

        if GSPSampleKey.gsp in self._data:
            axes[0, 0].plot(self._data[GSPSampleKey.gsp])
            axes[0, 0].set_title("GSP Generation")

        if "solar_azimuth" in self._data and "solar_elevation" in self._data:
                    axes[1, 1].plot(self._data["solar_azimuth"], label="Azimuth")
                    axes[1, 1].plot(self._data["solar_elevation"], label="Elevation")
                    axes[1, 1].set_title("Solar Position")
                    axes[1, 1].legend()

        if SatelliteSampleKey.satellite_actual in self._data:
            axes[1, 0].imshow(self._data[SatelliteSampleKey.satellite_actual])
            axes[1, 0].set_title("Satellite Data")

        plt.tight_layout()
        plt.show()
