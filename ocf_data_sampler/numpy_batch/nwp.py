"""Convert NWP to NumpyBatch"""

import numpy as np
import xarray as xr

from ocf_datapipes.batch import NWPBatchKey, NWPNumpyBatch
from ocf_datapipes.utils.utils import datetime64_to_float


def convert_nwp_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> NWPNumpyBatch:
    """Convert from Xarray to NWP NumpyBatch"""

    example: NWPNumpyBatch = {
        NWPBatchKey.nwp: da.values,
        NWPBatchKey.nwp_channel_names: da.channel.values,
        NWPBatchKey.nwp_init_time_utc: datetime64_to_float(da.init_time_utc.values),
        NWPBatchKey.nwp_step: (da.step.values / np.timedelta64(1, "h")).astype(np.int64),
    }

    if "target_time_utc" in da.coords:
        target_time = da.target_time_utc.values
        example[NWPBatchKey.nwp_target_time_utc] = datetime64_to_float(target_time)

    # TODO: Do we need this at all? Especially since it is only present in UKV data
    for batch_key, dataset_key in (
        (NWPBatchKey.nwp_y_osgb, "y_osgb"),
        (NWPBatchKey.nwp_x_osgb, "x_osgb"),
    ):
        if dataset_key in da.coords:
            example[batch_key] = da[dataset_key].values

    if t0_idx is not None:
        example[NWPBatchKey.nwp_t0_idx] = t0_idx

    return example
