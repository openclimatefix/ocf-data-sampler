"""Convert Satellite to NumpySample."""

import warnings

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.numpy_sample.converter import _convert_satellite


class SatelliteSampleKey:
    """Deprecated. Use dictionary keys in ``NumpySample`` instead."""

    satellite_actual = "satellite_actual"
    time_utc = "satellite_time_utc"
    x_geostationary = "satellite_x_geostationary"
    y_geostationary = "satellite_y_geostationary"
    t0_idx = "satellite_t0_idx"

    @staticmethod
    def __getattr__(name: str) -> str:
        """Warn on deprecated attribute access.

        Any attribute access is redirected to dictionary keys on ``NumpySample``.
        """
        warnings.warn(
            f"'{name}' is deprecated. Use the dictionary key in NumpySample instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattr__(name)


def convert_satellite_to_numpy_sample(
    da: xr.DataArray,
    t0_idx: int | None = None,
) -> NumpySample:
    """Convert from xarray DataArray to NumpySample.

    Args:
        da:
            Xarray DataArray containing satellite data.
        t0_idx:
            Index of the t0 timestamp in the time dimension of the satellite data.

    Returns:
        NumpySample:
            Dictionary-based numpy sample representation.
    """
    sample: NumpySample = {
        "metadata": {
            "t0_idx": t0_idx,
        },
    }

    # _convert_satellite mutates `sample` in-place
    _convert_satellite(da, sample)

    return sample
