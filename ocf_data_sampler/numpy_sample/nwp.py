"""Convert NWP to NumpySample"""

import pandas as pd
import xarray as xr


class NWPSampleKey:

    nwp = 'nwp'
    channel_names = 'nwp_channel_names'
    init_time_utc = 'nwp_init_time_utc'
    step = 'nwp_step'
    target_time_utc = 'nwp_target_time_utc'
    t0_idx = 'nwp_t0_idx'
    y_osgb = 'nwp_y_osgb'
    x_osgb = 'nwp_x_osgb'



def convert_nwp_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NWP NumpySample"""
    
    # Create example and add t if available
    sample = {
        NWPSampleKey.nwp: da.values,
        NWPSampleKey.channel_names: da.channel.values,
        NWPSampleKey.init_time_utc: da.init_time_utc.values.astype(float),
        NWPSampleKey.step: (da.step.values / pd.Timedelta("1h")).astype(int),
    }

    if "target_time_utc" in da.coords:
        sample[NWPSampleKey.target_time_utc] = da.target_time_utc.values.astype(float)

    # TODO: Do we need this at all? Especially since it is only present in UKV data
    for sample_key, dataset_key in ((NWPSampleKey.y_osgb, "y_osgb"),(NWPSampleKey.x_osgb, "x_osgb"),):
        if dataset_key in da.coords:
            sample[sample_key] = da[dataset_key].values

    if t0_idx is not None:
        sample[NWPSampleKey.t0_idx] = t0_idx
        
    return sample