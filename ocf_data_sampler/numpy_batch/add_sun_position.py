
import pvlib
import numpy as np
import pandas as pd
from ocf_datapipes.batch import BatchKey
from ocf_datapipes.utils.geospatial import osgb_to_lon_lat


def get_azimuth_and_elevation(
    datetimes: pd.DatetimeIndex, 
    lon: float, 
    lat: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the solar coordinates for multiple datetimes at a single location
    
    Args:
        datetimes: The datetimes to calculate for
        lon: The longitude
        lat: The latitude

    Returns:
        np.ndarray: The azimuth of the datetimes in degrees
        np.ndarray: The elevation of the datetimes in degrees
    """

    solpos = pvlib.solarposition.get_solarposition(
        time=datetimes,
        longitude=lon,
        latitude=lat,
        method='nrel_numpy'
    )
    azimuth = solpos["azimuth"].values
    elevation = solpos["elevation"].values
    return azimuth, elevation


def add_sun_position_to_numpy_batch(np_batch, modality_name: str):
    """
    Adds the standardized sun position to the NumpyBatch

        modality_name: Modality to add the sun position for
    """
    if modality_name not in {"gsp"}:
        raise NotImplementedError(f"Modality {modality_name} not supported")

    if modality_name == "gsp":
        times_utc: pd.DatetimeIndex = pd.to_datetime(np_batch[BatchKey.gsp_time_utc])
        x_osgb: float = np_batch[BatchKey.gsp_x_osgb] 
        y_osgb: float = np_batch[BatchKey.gsp_y_osgb] 
        lon, lat = osgb_to_lon_lat(x=x_osgb, y=y_osgb)
    
    azimuth, elevation = get_azimuth_and_elevation(times_utc, lon, lat)

    # Standardize

    # Azimuth is in range [0, 360] degrees
    azimuth = azimuth / 360

    # Elevation is in range [-90, 90] degrees
    elevation = elevation / 180 + 0.5
    
    # Store
    azimuth_batch_key = BatchKey[modality_name + "_solar_azimuth"]
    elevation_batch_key = BatchKey[modality_name + "_solar_elevation"]
    np_batch[azimuth_batch_key] = azimuth
    np_batch[elevation_batch_key] = elevation

    return np_batch