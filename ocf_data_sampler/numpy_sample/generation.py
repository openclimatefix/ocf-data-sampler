"""Convert Generation data to Numpy Sample."""

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.numpy_sample.converter import _convert_generation

# class GenerationSampleKey:
#     """Keys for the Generation sample dictionary."""

#     generation = "generation"
#     capacity_mwp = "capacity_mwp"
#     time_utc = "time_utc"
#     t0_idx = "t0_idx"
#     location_id = "location_id"
#     longitude = "longitude"
#     latitude = "latitude"


def convert_generation_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> NumpySample:
    """Convert from Xarray to NumpySample.

    Args:
        da: Xarray DataArray containing generation data
        t0_idx: Index of the t0 timestamp in the time dimension of the generation data
    """
    sample: NumpySample = {
        "metadata": {
            "t0_idx": t0_idx,
        }
    }

    _convert_generation(da, sample)

    return sample