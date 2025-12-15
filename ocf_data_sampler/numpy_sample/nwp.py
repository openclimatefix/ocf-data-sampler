"""Convert NWP to NumpySample."""

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.numpy_sample.converter import _convert_nwp

# class NWPSampleKey:
#     """Keys for NWP NumpySample."""

#     nwp = "nwp"
#     channel_names = "nwp_channel_names"
#     init_time_utc = "nwp_init_time_utc"
#     step = "nwp_step"
#     target_time_utc = "nwp_target_time_utc"
#     t0_idx = "nwp_t0_idx"


def convert_nwp_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> NumpySample:
    """Convert from Xarray to NWP NumpySample.

    Args:
        da: Xarray DataArray containing NWP data
        t0_idx: Index of the t0 timestamp in the time dimension of the NWP
    """
    sample : NumpySample= {
        "metadata": {
            "t0_idx": t0_idx,
        }
    } 
    _convert_nwp(da,sample)
    return sample
