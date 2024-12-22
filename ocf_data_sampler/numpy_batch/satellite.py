"""Convert Satellite to NumpyBatch"""
import xarray as xr


class SatelliteBatchKey:

    satellite_actual = 'satellite_actual'
    time_utc = 'satellite_time_utc'
    x_geostationary = 'satellite_x_geostationary'
    y_geostationary = 'satellite_y_geostationary'
    t0_idx = 'satellite_t0_idx'


def convert_satellite_to_numpy_batch(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpyBatch"""

    example = {
        SatelliteBatchKey.satellite_actual: da.values,
        SatelliteBatchKey.time_utc: da.time_utc.values.astype(float),
    }

    for batch_key, dataset_key in (
         (SatelliteBatchKey.x_geostationary, "x_geostationary"),
        (SatelliteBatchKey.y_geostationary, "y_geostationary"),
    ):
        example[batch_key] = da[dataset_key].values

    if t0_idx is not None:
        example[SatelliteBatchKey.t0_idx] = t0_idx

    return example
