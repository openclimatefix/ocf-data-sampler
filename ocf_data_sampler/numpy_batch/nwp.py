"""Convert NWP to NumpyBatch"""

import pandas as pd
import xarray as xr


class NWPBatchKey:

    nwp = 'nwp'
    nwp_channel_names = 'nwp_channel_names'
    nwp_init_time_utc = 'nwp_init_time_utc'
    nwp_step = 'nwp_step'
    nwp_target_time_utc = 'nwp_target_time_utc'
    nwp_t0_idx = 'nwp_t0_idx'
    nwp_y_osgb = 'nwp_y_osgb'
    nwp_x_osgb = 'nwp_x_osgb'


def convert_nwp_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NWP NumpyBatch"""

    example = {
        NWPBatchKey.nwp: da.values,
        NWPBatchKey.nwp_channel_names: da.channel.values,
        NWPBatchKey.nwp_init_time_utc: da.init_time_utc.values.astype(float),
        NWPBatchKey.nwp_step: (da.step.values / pd.Timedelta("1h")).astype(int),
    }

    if "target_time_utc" in da.coords:
        example[NWPBatchKey.nwp_target_time_utc] = da.target_time_utc.values.astype(float)

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
