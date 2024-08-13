"""Convert Satellite to NumpyBatch"""
import xarray as xr

from ocf_datapipes.batch import BatchKey, NumpyBatch


def convert_satellite_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> NumpyBatch:
    """Convert from Xarray to NumpyBatch"""
    example: NumpyBatch = {
        BatchKey.satellite_actual: da.values,
        BatchKey.satellite_time_utc: da.time_utc.values.astype(float),
    }

    for batch_key, dataset_key in (
         (BatchKey.satellite_x_geostationary, "x_geostationary"),
        (BatchKey.satellite_y_geostationary, "y_geostationary"),
    ):
        example[batch_key] = da[dataset_key].values

    if t0_idx is not None:
        example[BatchKey.satellite_t0_idx] = t0_idx

    return example