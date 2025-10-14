<<<<<<< HEAD:ocf_data_sampler/torch_datasets/sample/base.py
<<<<<<< HEAD:ocf_data_sampler/torch_datasets/sample/base.py
"""Sample class implementation for dataset handling and visualisation."""

import logging
=======
"""Functions to convert batches to tensors and move them to a given device."""
>>>>>>> origin/main:ocf_data_sampler/torch_datasets/utils/torch_batch_utils.py

import numpy as np
import torch

<<<<<<< HEAD:ocf_data_sampler/torch_datasets/sample/base.py
from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
)
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, NumpySample, TensorBatch
from ocf_data_sampler.torch_datasets.utils.validation_utils import (
    calculate_expected_shapes,
    check_dimensions,
    validation_warning,
)

logger = logging.getLogger(__name__)
=======
from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, TensorBatch

>>>>>>> origin/main:ocf_data_sampler/torch_datasets/utils/torch_batch_utils.py

def batch_to_tensor(batch: NumpyBatch) -> TensorBatch:
    """Recursively converts numpy arrays in nested dict to torch tensors.

    Args:
        batch: NumpyBatch with data in numpy arrays
    Returns:
        TensorBatch with data in torch tensors
    """
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = batch_to_tensor(v)
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                batch[k] = torch.tensor(v, dtype=torch.bool)
            elif np.issubdtype(v.dtype, np.number):
                batch[k] = torch.as_tensor(v)
    return batch


def copy_batch_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively copies tensors in nested dict to specified device.

    Args:
        batch: Nested dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to the new device
    """
    batch_copy = {}

    for k, v in batch.items():
        if isinstance(v, dict):
            batch_copy[k] = copy_batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch_copy[k] = v.to(device)
        else:
            batch_copy[k] = v
    return batch_copy

class Sample:
    """Handles PVNet sample operations."""

    def __init__(self, data: NumpySample) -> None:
        """Initialises UK Regional sample with data."""
        self._data = data

    def to_numpy(self) -> NumpySample:
        """Returns the data as a NumPy sample."""
        return self._data

    def save(self, path: str) -> None:
        """Saves sample to the specified path in pickle format."""
        # Saves to pickle format
        torch.save(self._data, path)

    @classmethod
    def load(cls, path: str) -> "Sample":
        """Loads sample from the specified path.

        Args:
            path: Path to the saved sample file.

        Returns:
            A UKRegionalSample instance with the loaded data.
        """
        # Loads from .pt format
        # TODO: We should move away from using torch.load(..., weights_only=False)
        return cls(torch.load(path, weights_only=False))

    # TODO make this work for both gsp and site
    def validate_sample(self, config: Configuration) -> dict:
        """Validates the sample, logging warnings and raising errors.

        Checks that the sample has the expected structure and data shapes based
        on the provided configuration. Critical issues (missing required data,
        shape mismatches) will raise a ValueError. Non-critical issues (e.g.,
        unexpected data components found) will be logged as warnings using
        the standard Python logging module.

        Args:
            config: Configuration object defining expected shapes and required fields.

        Returns:
            dict: A dictionary indicating success: `{"valid": True}`.
                  If validation fails due to a critical issue, an exception is raised
                  instead of returning. Warnings encountered are logged.

        Raises:
            TypeError: If `config` is not a Configuration object.
            ValueError: For critical validation failures like missing expected data,
                        incorrect data shapes, or missing required NWP providers.
        """
        if not isinstance(config, Configuration):
            raise TypeError("config must be Configuration object")

        # Calculate expected shapes from configuration
        expected_shapes = calculate_expected_shapes(config)

        # Check GSP shape if specified
        gsp_key = GSPSampleKey.gsp
        if gsp_key in expected_shapes and gsp_key not in self._data:
            raise ValueError(f"Configuration expects GSP data ('{gsp_key}') but is missing.")

        if gsp_key in self._data:
            if gsp_key in expected_shapes:
                gsp_data = self._data[gsp_key]
                check_dimensions(
                    actual_shape=gsp_data.shape,
                    expected_shape=expected_shapes[gsp_key],
                    name="GSP",
                )
            else:
                validation_warning(
                    message=f"GSP data ('{gsp_key}') is present but not expected in configuration.",
                    warning_type="unexpected_component",
                    component=str(gsp_key),
                )

        # Checks for NWP data
        nwp_key = NWPSampleKey.nwp
        if nwp_key in expected_shapes and nwp_key not in self._data:
            raise ValueError(f"Configuration expects NWP data ('{nwp_key}') but is missing.")

        if nwp_key in self._data:
            nwp_data_all_providers = self._data[nwp_key]
            if not isinstance(nwp_data_all_providers, dict):
                raise ValueError(f"NWP data ('{nwp_key}') should be a dictionary.")

            if nwp_key in expected_shapes:
                expected_providers = set(expected_shapes[nwp_key].keys())
                actual_providers = set(nwp_data_all_providers.keys())

                unexpected_providers = actual_providers - expected_providers
                if unexpected_providers:
                    validation_warning(
                        message=f"Unexpected NWP providers found: {list(unexpected_providers)}",
                        warning_type="unexpected_provider",
                        providers=list(unexpected_providers),
                    )

                missing_expected_providers = expected_providers - actual_providers
                if missing_expected_providers:
                    raise ValueError(
                        f"Expected NWP providers are missing from the data: "
                        f"{list(missing_expected_providers)}",
                    )

                for provider in expected_shapes[nwp_key]:
                    provider_data = nwp_data_all_providers[provider]

                    if "nwp" not in provider_data:
                        error_msg = (
                            f"Missing array key 'nwp' in NWP data for provider '{provider}'."
                        )
                        raise ValueError(error_msg)

                    nwp_array = provider_data["nwp"]
                    check_dimensions(
                        actual_shape=nwp_array.shape,
                        expected_shape=expected_shapes[nwp_key][provider],
                        name=f"NWP data ({provider})",
                    )
            else:
                validation_warning(
                    message=(
                        f"NWP data ('{nwp_key}') is present but not expected "
                        "in configuration."
                    ),
                    warning_type="unexpected_component",
                    component=str(nwp_key),
                )

        # Validate satellite data
        sat_key = SatelliteSampleKey.satellite_actual
        if sat_key in expected_shapes and sat_key not in self._data:
            raise ValueError(f"Configuration expects Satellite data ('{sat_key}') but is missing.")

        if sat_key in self._data:
            if sat_key in expected_shapes:
                sat_data = self._data[sat_key]
                check_dimensions(
                    actual_shape=sat_data.shape,
                    expected_shape=expected_shapes[sat_key],
                    name="Satellite data",
                )
            else:
                validation_warning(
                    message=(
                        f"Satellite data ('{sat_key}') is present but not expected "
                        "in configuration."
                    ),
                    warning_type="unexpected_component",
                    component=str(sat_key),
                )

        # Validate solar coordinates data
        solar_keys = ["solar_azimuth", "solar_elevation"]
        for solar_key in solar_keys:
            solar_name = solar_key.replace("_", " ").title()
            if solar_key in expected_shapes and solar_key not in self._data:
                raise ValueError(f"Configuration expects {solar_key} data but is missing.")

            if solar_key in self._data:
                if solar_key in expected_shapes:
                    solar_data = self._data[solar_key]
                    check_dimensions(
                        actual_shape=solar_data.shape,
                        expected_shape=expected_shapes[solar_key],
                        name=f"{solar_name} data",
                    )
                else:
                    validation_warning(
                        message=(
                            f"{solar_name} data is present but not expected "
                            "in configuration."
                        ),
                        warning_type="unexpected_component",
                        component=solar_key,
                    )

        # Check for potentially unexpected components
        checked_keys = {gsp_key, nwp_key, sat_key} | set(solar_keys)
        all_present_keys = set(self._data.keys())
        unexpected_present_keys = all_present_keys - set(expected_shapes.keys())

        for key in unexpected_present_keys:
            if key not in checked_keys:
                validation_warning(
                    message=(
                        f"Unexpected component '{key}' is present in data but not defined "
                        "in configuration's expected shapes."
                    ),
                    warning_type="unexpected_component",
                    component=str(key),
                )

        return {
            "valid": True,
        }

    # TODO make this work for both gsp and site
    def plot(self) -> None:
        """Plots the sample data for visualization."""
        from matplotlib import pyplot as plt

        _, axes = plt.subplots(2, 2, figsize=(12, 8))

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
