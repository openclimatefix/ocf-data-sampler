"""Convert NWP to NumpySample."""

import warnings

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.numpy_sample.converter import _convert_nwp


class NWPSampleKey:
    """Deprecated. Use dictionary keys in ``NumpySample`` instead."""

    nwp = "nwp"
    channel_names = "nwp_channel_names"
    init_time_utc = "nwp_init_time_utc"
    step = "nwp_step"
    target_time_utc = "nwp_target_time_utc"
    t0_idx = "nwp_t0_idx"

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


def convert_nwp_to_numpy_sample(
    da: xr.DataArray,
    t0_idx: int | None = None,
) -> NumpySample:
    """Convert from xarray DataArray to NWP NumpySample.

    Args:
        da:
            Xarray DataArray containing NWP data.
        t0_idx:
            Index of the t0 timestamp in the time dimension of the NWP data.

    Returns:
        NumpySample:
            Dictionary-based numpy sample representation.
    """
    sample: NumpySample = {
        "metadata": {
            "t0_idx": t0_idx,
        },
    }

    _convert_nwp(da, sample)

    return sample
