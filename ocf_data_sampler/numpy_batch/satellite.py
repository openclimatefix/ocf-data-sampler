"""Convert Satellite to NumpyBatch"""

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


def convert_satellite_to_numpy_batch(xr_data):
    example: NumpyBatch = {
        BatchKey.satellite_actual: xr_data.values,
        BatchKey.satellite_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.satellite_time_utc: datetime64_to_float(xr_data["time_utc"].values),
    }

    for batch_key, dataset_key in (
        (BatchKey.satellite_y_geostationary, "y_geostationary"),
        (BatchKey.satellite_x_geostationary, "x_geostationary"),
    ):
        example[batch_key] = xr_data[dataset_key].values

    return example