"""Convert site to Numpy Batch"""

import xarray as xr


class SiteBatchKey:

    generation = "site"
    site_capacity_kwp = "site_capacity_kwp"
    site_time_utc = "site_time_utc"
    site_t0_idx = "site_t0_idx"
    site_solar_azimuth = "site_solar_azimuth"
    site_solar_elevation = "site_solar_elevation"
    site_id = "site_id"


def convert_site_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpyBatch"""

    example = {
        SiteBatchKey.generation: da.values,
        SiteBatchKey.site_capacity_kwp: da.isel(time_utc=0)["capacity_kwp"].values,
        SiteBatchKey.site_time_utc: da["time_utc"].values.astype(float),
    }

    if t0_idx is not None:
        example[SiteBatchKey.site_t0_idx] = t0_idx

    return example
