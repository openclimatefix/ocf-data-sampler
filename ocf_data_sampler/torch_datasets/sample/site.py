"""PVNet Site sample implementation for netCDF data handling and conversion."""

import torch
from typing_extensions import override

from ocf_data_sampler.numpy_sample.common_types import NumpySample

from .base import SampleBase


# TODO this is now similar to the UKRegionalSample
# We should consider just having one Sample class for all datasets
class SiteSample(SampleBase):
    """Handles SiteSample specific operations."""

    def __init__(self, data: NumpySample) -> None:
        """Initializes the SiteSample object with the given NumpySample."""
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:
        return self._data

    @override
    def save(self, path: str) -> None:
        """Saves sample to the specified path in pickle format."""
        # Saves to pickle format
        torch.save(self._data, path)

    @classmethod
    @override
    def load(cls, path: str) -> "SiteSample":
        """Loads sample from the specified path.

        Args:
            path: Path to the saved sample file.

        Returns:
            A SiteSample instance with the loaded data.
        """
        # Loads from .pt format
        # TODO: We should move away from using torch.load(..., weights_only=False)
        return cls(torch.load(path, weights_only=False))

    @override
    def plot(self) -> None:
        # TODO - placeholder for now
        raise NotImplementedError("Plotting not yet implemented for SiteSample")
