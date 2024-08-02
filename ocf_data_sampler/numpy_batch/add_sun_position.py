
import pvlib
import numpy as np
import warnings
from ocf_datapipes.batch import BatchKey
from ocf_datapipes.utils.consts import (
    AZIMUTH_MEAN,
    AZIMUTH_STD,
    ELEVATION_MEAN,
    ELEVATION_STD,
)
from ocf_datapipes.utils.geospatial import osgb_to_lon_lat


def _get_azimuth_and_elevation(lon, lat, dt, must_be_finite):
    if type(dt[0]) == np.datetime64:
        # This caused an issue if it was 'datetime64[s]'
        dt = np.array(dt, dtype="datetime64[ns]")

    if not np.isfinite([lon, lat]).all():
        if must_be_finite:
            raise ValueError(f"Non-finite (lon, lat) = ({lon}, {lat}")
        return (
            np.full_like(dt, fill_value=np.nan).astype(np.float32),
            np.full_like(dt, fill_value=np.nan).astype(np.float32),
        )

    else:
        solpos = pvlib.solarposition.get_solarposition(
            time=dt,
            latitude=lat,
            longitude=lon,
            # Which `method` to use?
            # pyephem seemed to be a good mix between speed and ease
            # but causes segfaults!
            # nrel_numba doesn't work when using multiple worker processes.
            # nrel_c is probably fastest but requires C code to be
            #   manually compiled: https://midcdmz.nrel.gov/spa/
        )
        azimuth = solpos["azimuth"].values
        elevation = solpos["elevation"].values
        return azimuth, elevation


def add_sun_position(np_batch, modality_name: str):
    """
    Adds the sun position to the NumpyBatch

        modality_name: Modality to add the sun position for
    """
    assert modality_name in [
        "gsp",
    ], f"Cant add sun position on {modality_name}"

    if modality_name == "gsp":
        y_osgb: float = np_batch[BatchKey.gsp_y_osgb] 
        x_osgb: float = np_batch[BatchKey.gsp_x_osgb] 
        time_utc: np.ndarray = np_batch[BatchKey.gsp_time_utc]

    # As we move away from OSGB and towards lon, lat we can exclude more sources here
    if modality_name in ["gsp"]:
        # Convert to the units that pvlib expects: lon, lat
        lons, lats = osgb_to_lon_lat(x=x_osgb, y=y_osgb)

    # Elevations must be finite and non-nan except for PV data where values may be missing
    must_be_finite = modality_name != "pv"

    times = time_utc.astype("datetime64[s]")

    azimuth, elevation = _get_azimuth_and_elevation(lons, lats, times, must_be_finite)

    # Normalize
    azimuth = (azimuth - AZIMUTH_MEAN) / AZIMUTH_STD
    elevation = (elevation - ELEVATION_MEAN) / ELEVATION_STD

    # Store
    azimuth_batch_key = BatchKey[modality_name + "_solar_azimuth"]
    elevation_batch_key = BatchKey[modality_name + "_solar_elevation"]
    np_batch[azimuth_batch_key] = azimuth
    np_batch[elevation_batch_key] = elevation

    return np_batch