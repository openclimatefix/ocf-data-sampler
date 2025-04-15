"""Validate sample shape against expected shape - utility function."""

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.numpy_sample import GSPSampleKey, NWPSampleKey, SatelliteSampleKey


def check_dimensions(
    actual_shape: tuple[int, ...],
    expected_shape: tuple[int | None, ...],
    name: str,
) -> None:
    """Check if dimensions match between actual and expected shapes.

    Allows `None` in `expected_shape` to act as a wildcard for that dimension.

    Args:
        actual_shape: The actual shape of the data (e.g., array.shape).
        expected_shape: The expected shape, potentially with None as wildcards.
        name: Name of the data component for clear error messages.

    Raises:
        ValueError: If dimensions don't match according to the rules.
    """
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"'{name}' shape dimension mismatch: "
            f"Expected {len(expected_shape)} dimensions, got {len(actual_shape)} "
            f"(Actual shape: {actual_shape}, Expected shape: {expected_shape})",
        )

    zipped_dims = zip(actual_shape, expected_shape, strict=True)
    for i, (actual_dim, expected_dim) in enumerate(zipped_dims):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(f"{name} shape mismatch at dimension {i}")


def calculate_expected_shapes(
    config: Configuration,
) -> dict[str, tuple[int, ...]]:
    """Calculate expected shapes from configuration.

    Args:
        config: Configuration object or dictionary with shape information.
            If None, returns an empty dictionary.

    Returns:
        Dictionary mapping data keys to their expected shapes (tuples of integers).
    """
    expected_shapes = {}

    # Calculate expected shapes from input_data if available
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
            if NWPSampleKey.nwp not in expected_shapes:
                expected_shapes[NWPSampleKey.nwp] = {}

            # Add shapes for each provider directly
            for provider_key, provider_config in input_data.nwp.items():
                time_span = (
                    provider_config.interval_end_minutes -
                    provider_config.interval_start_minutes
                )

                resolution = provider_config.time_resolution_minutes
                num_timesteps = (time_span // resolution) + 1

                num_channels = len(provider_config.channels)
                height = provider_config.image_size_pixels_height
                width = provider_config.image_size_pixels_width

                # Store shape directly in nested dictionary
                expected_shapes[NWPSampleKey.nwp][provider_key] = (
                    num_timesteps,
                    num_channels,
                    height,
                    width,
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

    return expected_shapes
