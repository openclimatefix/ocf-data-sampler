"""Convert GSP to Numpy Sample"""

import xarray as xr


class GSPSampleKey:

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


def convert_gsp_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpySample"""

   # Extract values from the DataArray
    sample = {
        GSPSampleKey.gsp: da.values,
        GSPSampleKey.nominal_capacity_mwp: da.isel(time_utc=0)["nominal_capacity_mwp"].values,
        GSPSampleKey.effective_capacity_mwp: da.isel(time_utc=0)["effective_capacity_mwp"].values,
        GSPSampleKey.time_utc: da["time_utc"].values.astype(float),
    }

    if t0_idx is not None:
        sample[GSPSampleKey.t0_idx] = t0_idx

    return sample
