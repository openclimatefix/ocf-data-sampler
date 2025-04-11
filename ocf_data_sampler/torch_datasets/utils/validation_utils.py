"""Validate sample shape against expected shape - utility function."""

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
