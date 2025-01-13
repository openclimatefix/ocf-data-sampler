"""Convert site to Numpy Sample"""

import xarray as xr


class SiteSampleKey:

    generation = "site"
    capacity_kwp = "site_capacity_kwp"
    time_utc = "site_time_utc"
    t0_idx = "site_t0_idx"
    solar_azimuth = "site_solar_azimuth"
    solar_elevation = "site_solar_elevation"
    id = "site_id"


def convert_site_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpySample"""

    # Extract values from the DataArray
    example = {
        SiteSampleKey.generation: da.values,
        SiteSampleKey.capacity_kwp: da.isel(time_utc=0)["capacity_kwp"].values,
        SiteSampleKey.time_utc: da["time_utc"].values.astype(float),
    }

    if t0_idx is not None:
        example[SiteSampleKey.t0_idx] = t0_idx

    return example