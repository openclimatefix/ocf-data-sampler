"""Convert GSP to Numpy Batch"""

import xarray as xr


class GSPBatchKey:

    gsp = 'gsp'
    nominal_capacity_mwp = 'gsp_nominal_capacity_mwp'
    effective_capacity_mwp = 'gsp_effective_capacity_mwp'
    time_utc = 'gsp_time_utc'
    t0_idx = 'gsp_t0_idx'
    solar_azimuth = 'gsp_solar_azimuth'
    solar_elevation = 'gsp_solar_elevation'
    gsp_id = 'gsp_id'
    x_osgb = 'gsp_x_osgb'
    y_osgb = 'gsp_y_osgb'


def convert_gsp_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpyBatch"""

    example = {
        GSPBatchKey.gsp: da.values,
        GSPBatchKey.nominal_capacity_mwp: da.isel(time_utc=0)["nominal_capacity_mwp"].values,
        GSPBatchKey.effective_capacity_mwp: da.isel(time_utc=0)["effective_capacity_mwp"].values,
        GSPBatchKey.time_utc: da["time_utc"].values.astype(float),
    }

    if t0_idx is not None:
        example[GSPBatchKey.t0_idx] = t0_idx

    return example
