"""Validate sample shape against expected shape - utility function."""

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_sample import GSPSampleKey, NWPSampleKey, SatelliteSampleKey


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


def calculate_expected_shapes(
    config: Configuration,
) -> dict[str, tuple[int, ...]]:
    """Calculate expected shapes from configuration.

    Args:
        config: Configuration object with shape information.

    Returns:
        Dictionary mapping data keys to their expected shapes.
    """
    expected_shapes = {}
    input_data = config.input_data

    # Calculate GSP shape
    gsp_config = input_data.gsp
    expected_shapes[GSPSampleKey.gsp] = (
        _calculate_time_steps(
            gsp_config.interval_start_minutes,
            gsp_config.interval_end_minutes,
            gsp_config.time_resolution_minutes,
        ),
    )

    # Calculate NWP shape for multiple providers
    expected_shapes[NWPSampleKey.nwp] = {}
    for provider_key, provider_config in input_data.nwp.items():
        expected_shapes[NWPSampleKey.nwp][provider_key] = (
            _calculate_time_steps(
                provider_config.interval_start_minutes,
                provider_config.interval_end_minutes,
                provider_config.time_resolution_minutes,
            ),
            len(provider_config.channels),
            provider_config.image_size_pixels_height,
            provider_config.image_size_pixels_width,
        )

    # Calculate satellite shape
    sat_config = input_data.satellite
    expected_shapes[SatelliteSampleKey.satellite_actual] = (
        _calculate_time_steps(
            sat_config.interval_start_minutes,
            sat_config.interval_end_minutes,
            sat_config.time_resolution_minutes,
        ),
        len(sat_config.channels),
        sat_config.image_size_pixels_height,
        sat_config.image_size_pixels_width,
    )

    return expected_shapes


def _calculate_time_steps(start_minutes: int, end_minutes: int, resolution_minutes: int) -> int:
    """Calculate number of time steps based on interval and resolution.

    Args:
        start_minutes: Start of interval in minutes
        end_minutes: End of interval in minutes
        resolution_minutes: Time resolution in minutes

    Returns:
        Number of time steps
    """
    time_span = end_minutes - start_minutes
    return (time_span // resolution_minutes) + 1
