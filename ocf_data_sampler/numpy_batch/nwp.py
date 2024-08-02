"""Convert NWP to NumpyBatch"""

import numpy as np

from ocf_datapipes.batch import NWPBatchKey, NWPNumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


def convert_nwp_to_numpy_batch(xr_data):
    """Convert from Xarray to NWPBatchKey"""

    example: NWPNumpyBatch = {
        NWPBatchKey.nwp: xr_data.values,
        NWPBatchKey.nwp_t0_idx: xr_data.attrs["t0_idx"],
        NWPBatchKey.nwp_channel_names: xr_data.channel.values,
        NWPBatchKey.nwp_init_time_utc: datetime64_to_float(xr_data.init_time_utc.values),
        NWPBatchKey.nwp_step: (xr_data.step.values / np.timedelta64(1, "h")).astype(np.int64),
    }

    if "target_time_utc" in xr_data.coords:
        target_time = xr_data.target_time_utc.values
        example[NWPBatchKey.nwp_target_time_utc] = datetime64_to_float(target_time)

    for batch_key, dataset_key in (
        (NWPBatchKey.nwp_y_osgb, "y_osgb"),
        (NWPBatchKey.nwp_x_osgb, "x_osgb"),
    ):
        if dataset_key in xr_data.coords:
            example[batch_key] = xr_data[dataset_key].values

    return example
