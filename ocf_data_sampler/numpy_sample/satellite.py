"""Convert Satellite to NumpySample."""

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.numpy_sample.converter import _convert_satellite

# class SatelliteSampleKey:
#     """Keys for the SatelliteSample dictionary."""

#     satellite_actual = "satellite_actual"
#     time_utc = "satellite_time_utc"
#     x_geostationary = "satellite_x_geostationary"
#     y_geostationary = "satellite_y_geostationary"
#     t0_idx = "satellite_t0_idx"


def convert_satellite_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> NumpySample:
    """Convert from Xarray to NumpySample.

    Args:
        da: xarray DataArray containing satellite data
        t0_idx: Index of the t0 timestamp in the time dimension of the satellite data
    """
    sample = {
        "metadata" : {
            "t0_idx" : t0_idx
        }
    }

    sample = _convert_satellite(da,sample)
    return sample
