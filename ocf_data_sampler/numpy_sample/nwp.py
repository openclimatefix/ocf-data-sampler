"""Convert NWP to NumpySample."""

import pandas as pd
import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample


class NWPSampleKey:
    """Keys for NWP NumpySample."""

    nwp = "nwp"
    channel_names = "nwp_channel_names"
    init_time_utc = "nwp_init_time_utc"
    step = "nwp_step"
    target_time_utc = "nwp_target_time_utc"
    t0_idx = "nwp_t0_idx"


def convert_nwp_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> NumpySample:
    """Convert from Xarray to NWP NumpySample.

    Args:
        da: Xarray DataArray containing NWP data
        t0_idx: Index of the t0 timestamp in the time dimension of the NWP
    """
    sample = {
        NWPSampleKey.nwp: da.values,
        NWPSampleKey.channel_names: da.channel.values,
        NWPSampleKey.init_time_utc: da.init_time_utc.values.astype(float),
        NWPSampleKey.step: (da.step.values / pd.Timedelta("1h")).astype(int),
        NWPSampleKey.target_time_utc: da.target_time_utc.values.astype(float),
    }

    if t0_idx is not None:
        sample[NWPSampleKey.t0_idx] = t0_idx

    return sample
