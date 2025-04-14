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
from ocf_data_sampler.torch_datasets.utils.validation_utils import check_dimensions


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

    def validate_sample(self, config: Configuration | None = None) -> dict:
        """Validates that the sample has the expected structure and data shapes.

        Args:
            config: Optional configuration dict with expected shapes and required fields.
                If None, uses default validation rules.

        Returns:
            dict: Validation results with status and any validation errors.
        """

        # Check for required keys
        for key in config.get("required_keys", []):
            if key not in self._data:
                raise ValueError(f"Missing required key: {key}")

        # Start with configured expected shapes
        expected_shapes = {}
        if hasattr(config, "expected_shapes"):
            expected_shapes.update(config.expected_shapes)
        elif isinstance(config, dict) and "expected_shapes" in config:
            expected_shapes.update(config["expected_shapes"])

        # Calculate expected shapes from config if available
        if hasattr(config, "input_data"):
            input_data = config.input_data

            # Calculate GSP shape
            if hasattr(input_data, "gsp") and input_data.gsp is not None:
                gsp_config = input_data.gsp
                time_span = (
                    gsp_config.interval_end_minutes -
                    gsp_config.interval_start_minutes
                )
                resolution = gsp_config.time_resolution_minutes
                expected_length = (time_span // resolution) + 1
                expected_shapes[GSPSampleKey.gsp] = (expected_length,)

            # Calculate NWP shape
            if hasattr(input_data, "nwp") and input_data.nwp is not None:
                for provider in input_data.nwp.values():
                    expected_shapes[NWPSampleKey.nwp] = (
                        provider.image_size_pixels_height,
                        provider.image_size_pixels_width,
                    )

            # Calculate satellite shape
            if hasattr(input_data, "satellite") and input_data.satellite is not None:
                sat_config = input_data.satellite
                channels = len(sat_config.channels)
                time_span = (
                    sat_config.interval_end_minutes -
                    sat_config.interval_start_minutes
                )
                resolution = sat_config.time_resolution_minutes
                time_steps = (time_span // resolution) + 1

                expected_shapes[SatelliteSampleKey.satellite_actual] = (
                    time_steps,
                    channels,
                    sat_config.image_size_pixels_height,
                    sat_config.image_size_pixels_width,
                )

        # Check GSP shape if specified
        if GSPSampleKey.gsp in self._data:
            gsp_data = self._data[GSPSampleKey.gsp]
            
            # Check expected shape of GSP data if configured
            if GSPSampleKey.gsp in expected_shapes:
                check_dimensions(
                    actual_shape=gsp_data.shape,
                    expected_shape=expected_shapes[GSPSampleKey.gsp],
                    name="GSP",
                )

        # Checks for NWP data - nested structure
        if NWPSampleKey.nwp in self._data:
            nwp_data = self._data[NWPSampleKey.nwp]
            if not isinstance(nwp_data, dict):
                raise ValueError("NWP data should be a dictionary")

            # Validate NWP data structure for each provider
            for provider, provider_data in nwp_data.items():
                if "nwp" not in provider_data:
                    raise ValueError(f"Missing 'nwp' key in NWP data for {provider}")

                # Check expected shape of NWP data if configured
                if NWPSampleKey.nwp in expected_shapes and "nwp" in provider_data:
                    nwp_array = provider_data["nwp"]
                    actual_shape = nwp_array.shape
                    expected_shape = expected_shapes[NWPSampleKey.nwp]

                    check_dimensions(
                        actual_shape=actual_shape,
                        expected_shape=expected_shape,
                        name=f"NWP spatial ({provider})",
                    )

        # Validate satellite data
        if SatelliteSampleKey.satellite_actual in self._data:
            sat_data = self._data[SatelliteSampleKey.satellite_actual]

            if len(sat_data.shape) < 2:
                raise ValueError(
                    f"Satellite data should have at least 2 dimensions, got shape {sat_data.shape}",
                )

            if SatelliteSampleKey.satellite_actual in expected_shapes:
                sat_data = self._data[SatelliteSampleKey.satellite_actual]
                actual_spatial_dims = sat_data.shape[-2:]
                expected_spatial_dims = expected_shapes[SatelliteSampleKey.satellite_actual][-2:]

                check_dimensions(
                    actual_shape=tuple(actual_spatial_dims),
                    expected_shape=expected_spatial_dims,
                    name="Satellite spatial",
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
