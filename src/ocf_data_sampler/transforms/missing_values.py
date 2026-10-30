"""Validation and filling of missing values."""

import numpy as np

from ocf_data_sampler.common.types import DataArrayLike


def validate_and_fill_nans(
    da: DataArrayLike,
    fill_value: float | None,
    *,
    reject_nan: bool = False,
    reject_inf: bool = True,
    source: str,
) -> None:
    """Reject disallowed input values and optionally fill NaNs.

    Replaces `.data` only when filling NaNs, without modifying the original NumPy array.

    Args:
        da: Materialised DataArray-like to fill and validate.
        fill_value: Finite replacement for NaNs, or None to preserve them if permitted.
            Must be None when reject_nan is True.
        reject_nan: Whether NaNs in the input raise an error.
        reject_inf: Whether positive and negative infinity in the input raise an error.
        source: Source identifier included in validation errors.

    Raises:
        ValueError: A fill value is configured with reject_nan, or the input contains
            disallowed NaNs or infinities.
    """
    if reject_nan and fill_value is not None:
        raise ValueError(f"{source}: fill_value must be None when reject_nan is True")

    if reject_inf and np.isinf(da.data).any():
        raise ValueError(f"{source} contains infinities")

    if (fill_value is not None) or reject_nan:
        nan_mask = np.isnan(da.data)
        contains_nans = nan_mask.any()

        if contains_nans:
            if reject_nan:
                raise ValueError(f"{source} contains NaNs")
            if fill_value is not None:
                da.data = np.where(nan_mask, fill_value, da.data)
