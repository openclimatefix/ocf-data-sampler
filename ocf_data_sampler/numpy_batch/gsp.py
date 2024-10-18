"""Convert GSP to Numpy Batch"""

import xarray as xr


class GSPBatchKey:

    gsp = 'gsp'
    gsp_nominal_capacity_mwp = 'gsp_nominal_capacity_mwp'
    gsp_effective_capacity_mwp = 'gsp_effective_capacity_mwp'
    gsp_time_utc = 'gsp_time_utc'
    gsp_t0_idx = 'gsp_t0_idx'


def convert_gsp_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpyBatch"""

    example = {
        GSPBatchKey.gsp: da.values,
        GSPBatchKey.gsp_nominal_capacity_mwp: da.isel(time_utc=0)["nominal_capacity_mwp"].values,
        GSPBatchKey.gsp_effective_capacity_mwp: da.isel(time_utc=0)["effective_capacity_mwp"].values,
        GSPBatchKey.gsp_time_utc: da["time_utc"].values.astype(float),
    }

    if t0_idx is not None:
        example[GSPBatchKey.gsp_t0_idx] = t0_idx

    return example
