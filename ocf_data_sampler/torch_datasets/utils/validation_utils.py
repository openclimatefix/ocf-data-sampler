"""Validate sample shape against expected shape - utility function."""

import logging
from typing import Any

from ocf_data_sampler.config.model import Configuration, TimeWindowMixin

logger = logging.getLogger(__name__)


def check_dimensions(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    """Check if dimensions match between actual and expected shapes.

    Args:
        actual_shape: The actual shape of the data (e.g., array.shape).
        expected_shape: The expected shape.
        name: Name of the data component for clear error messages.

    Raises:
        ValueError: If dimensions don't match.
    """
    if actual_shape != expected_shape:
        raise ValueError(
            f"'{name}' shape mismatch: "
            f"Actual shape: {actual_shape}, Expected shape: {expected_shape}",
        )


def calculate_expected_shapes(config: Configuration) -> dict[str, tuple[int, ...]]:
    """Calculate expected shapes from configuration.

    Args:
        config: Configuration object with shape information.

    Returns:
        Dictionary mapping data keys to their expected shapes.
    """
    expected_shapes = {}
    input_data = config.input_data

    # Calculate generation shape
    expected_shapes["generation"] = (get_num_time_steps(input_data.generation),)

    # Calculate NWP shape for multiple providers
    for provider_key, provider_config in input_data.nwp.items():
        expected_shapes[f"nwp_{provider_key}"] = (
            get_num_time_steps(provider_config),
            len(provider_config.channels),
            provider_config.image_size_pixels_height,
            provider_config.image_size_pixels_width,
        )

    # Calculate satellite shape
    sat_config = input_data.satellite
    expected_shapes["satellite"] = (
        get_num_time_steps(sat_config),
        len(sat_config.channels),
        sat_config.image_size_pixels_height,
        sat_config.image_size_pixels_width,
    )

    # Calculate solar coordinates shapes
    expected_shapes["solar_azimuth"] = (get_num_time_steps(input_data.solar_position),)
    expected_shapes["solar_elevation"] = expected_shapes["solar_azimuth"]

    return expected_shapes


def validation_warning(
    message: str,
    warning_type: str,
    *,
    component: str | None = None,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Constructs warning details and logs a standard warning message.

    Args:
        message: The base warning message string.
        warning_type: The category of the warning (e.g., 'unexpected_component').
        component: Optional component identifier (e.g., 'generation').
        providers: Optional list of provider names (e.g., ['ukv']).

    Returns:
        None - This function now directly logs the warning.
    """
    warning_info: dict[str, Any] = {"type": warning_type, "message": message}
    log_message_parts = [message]
    log_message_parts.append(f"(Type: {warning_type}")

    if component is not None:
        warning_info["component"] = component
        log_message_parts.append(f", Component: {component}")
    if providers is not None:
        warning_info["providers"] = providers
        log_message_parts.append(f", Providers: {providers}")

    log_message_parts.append(")")
    log_message = " ".join(log_message_parts)
    logger.warning(log_message)


def get_num_time_steps(source_config: TimeWindowMixin) -> int:
    """Calculate number of time steps based on config interval and resolution.

    Args:
        source_config: Config for a single data source

    Returns:
        Number of time steps
    """
    return (
        source_config.interval_end_minutes - source_config.interval_start_minutes
    ) // source_config.time_resolution_minutes + 1
