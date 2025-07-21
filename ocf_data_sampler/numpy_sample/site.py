"""Convert site to Numpy Sample."""

import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample


class SiteSampleKey:
    """Keys for the site sample dictionary."""

    generation = "site"
    capacity_kwp = "site_capacity_kwp"
    time_utc = "site_time_utc"
    t0_idx = "site_t0_idx"
    id = "site_id"



def convert_site_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> NumpySample:
    """Convert from Xarray to NumpySample.

    Args:
        da: xarray DataArray containing site data
        t0_idx: Index of the t0 timestamp in the time dimension of the site data
    """
    sample = {
        SiteSampleKey.generation: da.values,
        SiteSampleKey.capacity_kwp: da.isel(time_utc=0)["capacity_kwp"].values,
        SiteSampleKey.time_utc: da["time_utc"].values.astype(float),
        SiteSampleKey.id: da["site_id"].values,
    }

    if t0_idx is not None:
        sample[SiteSampleKey.t0_idx] = t0_idx

    return sample
