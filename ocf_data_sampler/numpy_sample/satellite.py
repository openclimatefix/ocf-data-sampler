"""Convert Satellite to NumpySample"""
import xarray as xr


class SatelliteSampleKey:

    satellite_actual = 'satellite_actual'
    time_utc = 'satellite_time_utc'
    x_geostationary = 'satellite_x_geostationary'
    y_geostationary = 'satellite_y_geostationary'
    t0_idx = 'satellite_t0_idx'


def convert_satellite_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpySample"""
    sample = {
        SatelliteSampleKey.satellite_actual: da.values,
        SatelliteSampleKey.time_utc: da.time_utc.values.astype(float),
    }

    for sample_key, dataset_key in (
         (SatelliteSampleKey.x_geostationary, "x_geostationary"),
        (SatelliteSampleKey.y_geostationary, "y_geostationary"),
    ):
        sample[sample_key] = da[dataset_key].values

    if t0_idx is not None:
        sample[SatelliteSampleKey.t0_idx] = t0_idx

    return sample