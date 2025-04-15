"""PVNet UK Regional sample implementation for dataset handling and visualisation."""

import torch
from typing_extensions import override

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
)
from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.torch_datasets.sample.base import SampleBase
from ocf_data_sampler.torch_datasets.utils.validation_utils import (
    calculate_expected_shapes,
    check_dimensions,
)


class UKRegionalSample(SampleBase):
    """Handles UK Regional PVNet data operations."""

    def __init__(self, data: NumpySample) -> None:
        """Initialises UK Regional sample with data."""
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:
        """Returns the data as a NumPy sample."""
        return self._data

    @override
    def save(self, path: str) -> None:
        """Saves sample to the specified path in pickle format."""
        # Saves to pickle format
        torch.save(self._data, path)

    @classmethod
    @override
    def load(cls, path: str) -> "UKRegionalSample":
        """Loads sample from the specified path.

        Args:
            path: Path to the saved sample file.

        Returns:
            A UKRegionalSample instance with the loaded data.
        """
        # Loads from .pt format
        # TODO: We should move away from using torch.load(..., weights_only=False)
        return cls(torch.load(path, weights_only=False))

    def validate_sample(self, config: Configuration) -> bool:
        """Validates that the sample has the expected structure and data shapes.

        Args:
            config: Configuration dict with expected shapes and required fields.

        Returns:
            bool: True if validation passes, otherwise raises an exception.
        """
        if not isinstance(config, Configuration):
            raise TypeError("config must be Configuration object")

        # Calculate expected shapes from configuration
        expected_shapes = calculate_expected_shapes(config)

        # Check GSP shape if specified
        gsp_key = GSPSampleKey.gsp
        # Check if GSP data is expected but missing
        if gsp_key in expected_shapes and gsp_key not in self._data:
            raise ValueError(f"Configuration expects GSP data ('{gsp_key}') but is missing.")

        # Check GSP shape if data exists and is expected
        if gsp_key in self._data and gsp_key in expected_shapes:
            gsp_data = self._data[gsp_key]
            check_dimensions(
                actual_shape=gsp_data.shape,
                expected_shape=expected_shapes[gsp_key],
                name="GSP",
            )

        # Checks for NWP data - nested structure
        nwp_key = NWPSampleKey.nwp
        if nwp_key in expected_shapes and expected_shapes[nwp_key] and nwp_key not in self._data:
             raise ValueError(f"Configuration expects NWP data ('{nwp_key}') but is missing.")

        # Check NWP structure and shapes if data exists
        if nwp_key in self._data:
            nwp_data_all_providers = self._data[nwp_key]
            if not isinstance(nwp_data_all_providers, dict):
                raise ValueError(f"NWP data ('{nwp_key}') should be a dictionary.")

            # Loop through providers present in actual data
            for provider, provider_data in nwp_data_all_providers.items():
                if "nwp" not in provider_data:
                    raise ValueError(f"Missing array key in NWP data for provider '{provider}'.")

                if nwp_key in expected_shapes and provider in expected_shapes[nwp_key]:
                    nwp_array = provider_data["nwp"]
                    actual_shape = nwp_array.shape
                    expected_shape = expected_shapes[nwp_key][provider]

                    check_dimensions(
                        actual_shape=actual_shape,
                        expected_shape=expected_shape,
                        name=f"NWP data ({provider})",
                    )

        # Validate satellite data
        sat_key = SatelliteSampleKey.satellite_actual
        # Check if Satellite data is expected but missing
        if sat_key in expected_shapes and sat_key not in self._data:
            raise ValueError(f"Configuration expects Satellite data ('{sat_key}') but is missing.")

        # Check satellite shape if data exists and is expected
        if sat_key in self._data and sat_key in expected_shapes:
            sat_data = self._data[sat_key]
            check_dimensions(
                actual_shape=sat_data.shape,
                expected_shape=expected_shapes[sat_key],
                name="Satellite data",
            )

        return True

    @override
    def plot(self) -> None:
        """Plots the sample data for visualization."""
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
