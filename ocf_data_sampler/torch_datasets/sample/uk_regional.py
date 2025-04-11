"""PVNet UK Regional sample implementation for dataset handling and visualisation."""

import logging

import torch
from typing_extensions import override

from ocf_data_sampler.config.load import load_yaml_configuration
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
        validation_result = {
            "valid": True,
            "errors": [],
        }

        # Default config if none
        if config is None:
            config = {
                "required_keys": [
                    GSPSampleKey.gsp,
                    NWPSampleKey.nwp,
                    SatelliteSampleKey.satellite_actual,
                ],
                "expected_shapes": {
                    GSPSampleKey.gsp: (7,),
                },
            }

        # Check for required keys
        for key in config.get("required_keys", []):
            if key not in self._data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required key: {key}")

        # Check data shapes
        for key, expected_shape in config.get("expected_shapes", {}).items():
            if key in self._data:
                actual_shape = self._data[key].shape
                # Check if shapes match
                shape_valid = True
                if len(actual_shape) != len(expected_shape):
                    shape_valid = False
                else:
                    for actual_dim, expected_dim in zip(actual_shape, expected_shape, strict=False):
                        if expected_dim is not None and actual_dim != expected_dim:
                            shape_valid = False
                            break

                if not shape_valid:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}",
                    )

        # Checks for NWP data - nested structure
        if NWPSampleKey.nwp in self._data:
            nwp_data = self._data[NWPSampleKey.nwp]
            if not isinstance(nwp_data, dict):
                validation_result["valid"] = False
                validation_result["errors"].append("NWP data should be a dictionary")
            else:
                # Validate NWP data structure for each provider (ukv, ecmwf)
                for provider, provider_data in nwp_data.items():
                    if "nwp" not in provider_data:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Missing 'nwp' key in NWP data for {provider}",
                        )

                    # Check expected shape of NWP data if configured
                    if config.get("nwp_shape") and "nwp" in provider_data:
                        nwp_array = provider_data["nwp"]
                        spatial_dims = nwp_array.shape[-2:]

                        if tuple(spatial_dims) != config.get("nwp_shape"):
                            validation_result["valid"] = False
                            validation_result["errors"].append(
                                f"NWP shape mismatch for {provider}: "
                                f"expected {config.get('nwp_shape')} for spatial dimensions, "
                                f"got {spatial_dims}",
                            )

        # Add satellite validation
        if SatelliteSampleKey.satellite_actual in self._data:
            sat_data = self._data[SatelliteSampleKey.satellite_actual]

            if len(sat_data.shape) >= 2:
                spatial_dims = sat_data.shape[-2:]

                # Check specific satellite shape if in config
                if config.get("satellite_shape"):
                    expected_spatial_dims = config.get("satellite_shape")[-2:]
                    if spatial_dims != expected_spatial_dims:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Satellite spatial dimensions mismatch: "
                            f"expected {expected_spatial_dims}, got {spatial_dims}",
                        )
            else:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"Satellite data should have at least 2 dimensions, got shape {sat_data.shape}",
                )

        return validation_result

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


def validate_samples(
    samples: list[UKRegionalSample],
    config_or_path: dict[str, object] | str | None = None,
) -> dict[str, object]:
    """Validates a collection of UKRegionalSample objects.

    Args:
        samples: List of UKRegionalSample objects
        config_or_path: Either a configuration dictionary or path to a validation config file

    Returns:
        dict: Summary of validation results
    """
    # Determine if path or already a config dict
    config = None
    if config_or_path is not None:
        if isinstance(config_or_path, dict):
            config = config_or_path
        else:
            try:
                config_obj = load_yaml_configuration(config_or_path)
                config = config_obj.dict()
            except Exception as e:
                logging.warning(f"Failed to load config from {config_or_path}: {e}")

    results = {
        "total_samples": len(samples),
        "valid_samples": 0,
        "invalid_samples": 0,
        "error_summary": {},
    }

    for i, sample in enumerate(samples):
        validation = sample.validate_sample(config)
        if validation["valid"]:
            results["valid_samples"] += 1
        else:
            results["invalid_samples"] += 1
            for error in validation["errors"]:
                if error not in results["error_summary"]:
                    results["error_summary"][error] = []
                results["error_summary"][error].append(i)

    return results
