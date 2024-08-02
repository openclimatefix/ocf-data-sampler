"""Convert GSP to Numpy Batch"""

from ocf_datapipes.batch import BatchKey, NumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float



def convert_gsp_to_numpy_batch(xr_data):
    """Convert from Xarray to NumpyBatch"""

    example: NumpyBatch = {
        BatchKey.gsp: xr_data.values,
        BatchKey.gsp_t0_idx: xr_data.attrs["t0_idx"],
        BatchKey.gsp_id: xr_data.gsp_id.values,
        BatchKey.gsp_nominal_capacity_mwp: xr_data.isel(time_utc=0)["nominal_capacity_mwp"].values,
        BatchKey.gsp_effective_capacity_mwp: (
            xr_data.isel(time_utc=0)["effective_capacity_mwp"].values
        ),
        BatchKey.gsp_time_utc: datetime64_to_float(xr_data["time_utc"].values),
    }

    # Coordinates
    for batch_key, dataset_key in (
        (BatchKey.gsp_y_osgb, "y_osgb"),
        (BatchKey.gsp_x_osgb, "x_osgb"),
    ):
        if dataset_key in xr_data.coords.keys():
            example[batch_key] = xr_data[dataset_key].item()

    return example
