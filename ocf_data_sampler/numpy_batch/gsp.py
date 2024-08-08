"""Convert GSP to Numpy Batch"""

import xarray as xr
from ocf_datapipes.batch import BatchKey, NumpyBatch


def convert_gsp_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> NumpyBatch:
    """Convert from Xarray to NumpyBatch"""

    example: NumpyBatch = {
        BatchKey.gsp: da.values,
        BatchKey.gsp_id: da.gsp_id.values,
        BatchKey.gsp_nominal_capacity_mwp: da.isel(time_utc=0)["nominal_capacity_mwp"].values,
        BatchKey.gsp_effective_capacity_mwp: da.isel(time_utc=0)["effective_capacity_mwp"].values,
        BatchKey.gsp_time_utc: da["time_utc"].values.astype(float),
        BatchKey.gsp_x_osgb: da.x_osgb.item(),
        BatchKey.gsp_y_osgb: da.y_osgb.item(),
    }

    if t0_idx is not None:
        example[BatchKey.gsp_t0_idx] = t0_idx

    return example
