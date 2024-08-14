
import pvlib
import numpy as np
import pandas as pd
from ocf_datapipes.batch import BatchKey, NumpyBatch


def calculate_azimuth_and_elevation(
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


def make_sun_position_numpy_batch(
        datetimes: pd.DatetimeIndex, 
        lon: float, 
        lat: float, 
        key_preffix: str = "gsp"
) -> NumpyBatch:
    """Creates NumpyBatch with standardized solar coordinates

    Args:
        datetimes: The datetimes to calculate solar angles for
        lon: The longitude
        lat: The latitude
    """
    
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Normalise

    # Azimuth is in range [0, 360] degrees
    azimuth = azimuth / 360

    # Elevation is in range [-90, 90] degrees
    elevation = elevation / 180 + 0.5
    
    # Make NumpyBatch
    sun_numpy_batch: NumpyBatch = {
        BatchKey[key_preffix + "_solar_azimuth"]: azimuth,
        BatchKey[key_preffix + "_solar_elevation"]: elevation,
    }

    return sun_numpy_batch