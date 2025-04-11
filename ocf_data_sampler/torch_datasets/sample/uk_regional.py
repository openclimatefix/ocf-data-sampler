"""PVNet UK Regional sample implementation for dataset handling and visualisation."""


import torch
from typing_extensions import override

from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
)
from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.torch_datasets.sample.base import SampleBase


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

    def validate_sample(self, config: dict | None = None) -> dict:
        """Validates that the sample has the expected structure and data shapes.

        Args:
            config: Optional configuration dict with expected shapes and required fields.
                If None, uses default validation rules.

        Returns:
            dict: Validation results with status and any validation errors.
        """
        # Default config if none
        if config is None:
            config = {
                "required_keys": [
                    GSPSampleKey.gsp,
                    NWPSampleKey.nwp,
                    SatelliteSampleKey.satellite_actual,
                ],
            }

        # Check for required keys
        for key in config.get("required_keys", []):
            if key not in self._data:
                raise ValueError(f"Missing required key: {key}")

        expected_shapes = config.get("expected_shapes", {}).copy()

        # Calculate expected shapes from config if available
        if "input_data" in config:
            if GSPSampleKey.gsp not in expected_shapes and "gsp" in config["input_data"]:
                gsp_config = config["input_data"]["gsp"]
                time_span = (
                    gsp_config["interval_end_minutes"] -
                    gsp_config["interval_start_minutes"]
                )
                resolution = gsp_config["time_resolution_minutes"]
                expected_length = (time_span // resolution) + 1
                expected_shapes[GSPSampleKey.gsp] = (expected_length,)

            # Calculate NWP shape
            if "nwp" in config["input_data"]:
                for provider in config["input_data"]["nwp"].values():
                    if isinstance(provider, dict) and "image_size_pixels_height" in provider:
                        config["nwp_shape"] = (
                            provider["image_size_pixels_height"],
                            provider["image_size_pixels_width"],
                        )
                        break

            # Calculate satellite shape
            if "satellite" in config["input_data"]:
                sat_config = config["input_data"]["satellite"]
                has_height = "image_size_pixels_height" in sat_config
                has_width = "image_size_pixels_width" in sat_config
                if has_height and has_width:
                    channels = len(sat_config.get("channels", []))
                    interval_end = sat_config["interval_end_minutes"]
                    interval_start = sat_config["interval_start_minutes"]
                    time_span = interval_end - interval_start
                    resolution = sat_config["time_resolution_minutes"]
                    time_steps = (time_span // resolution) + 1

                    satellite_shape = (
                        time_steps,
                        channels,
                        sat_config["image_size_pixels_height"],
                        sat_config["image_size_pixels_width"],
                    )
                    config["satellite_shape"] = satellite_shape

        # Check data shapes against calculated or config given shapes
        for key, expected_shape in expected_shapes.items():
            if key in self._data:
                actual_shape = self._data[key].shape

                # Check dimensions match
                if len(actual_shape) != len(expected_shape):
                    raise ValueError(
                        f"Shape dimension mismatch for {key}: "
                        f"expected {len(expected_shape)} dimensions, got {len(actual_shape)}",
                    )

                # Check each dimension matches
                zipped_dims = zip(actual_shape, expected_shape, strict=False)
                for i, (actual_dim, expected_dim) in enumerate(zipped_dims):
                    if expected_dim is not None and actual_dim != expected_dim:
                        raise ValueError(
                            f"Shape mismatch for {key} at dimension {i}: "
                            f"expected {expected_dim}, got {actual_dim}",
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
                if config.get("nwp_shape") and "nwp" in provider_data:
                    nwp_array = provider_data["nwp"]
                    spatial_dims = nwp_array.shape[-2:]

                    if tuple(spatial_dims) != config.get("nwp_shape"):
                        raise ValueError(
                            f"NWP shape mismatch for {provider}: "
                            f"expected {config.get('nwp_shape')} for spatial dimensions, "
                            f"got {spatial_dims}",
                        )

        # Validate satellite data
        if SatelliteSampleKey.satellite_actual in self._data:
            sat_data = self._data[SatelliteSampleKey.satellite_actual]

            if len(sat_data.shape) < 2:
                raise ValueError(
                    f"Satellite data should have at least 2 dimensions, got shape {sat_data.shape}",
                )

            spatial_dims = sat_data.shape[-2:]
            if config.get("satellite_shape"):
                expected_spatial_dims = config.get("satellite_shape")[-2:]
                if spatial_dims != expected_spatial_dims:
                    raise ValueError(
                        f"Satellite spatial dimensions mismatch: "
                        f"expected {expected_spatial_dims}, got {spatial_dims}",
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
