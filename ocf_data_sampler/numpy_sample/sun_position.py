"""Module for calculating solar position."""

import numpy as np
import pandas as pd
import pvlib

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def calculate_azimuth_and_elevation(
    datetimes: pd.DatetimeIndex,
    lon: float,
    lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the solar coordinates for multiple datetimes at a single location.

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
        method="nrel_numpy",
    )

    return solpos["azimuth"].values, solpos["elevation"].values


def make_sun_position_numpy_sample(
    datetimes: pd.DatetimeIndex,
    lon: float,
    lat: float,
) -> NumpySample:
    """Creates NumpySample with standardized solar coordinates.

    Args:
        datetimes: The datetimes to calculate solar angles for
        lon: The longitude
        lat: The latitude
    """
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Normalise
    # Azimuth is in range [0, 360] degrees
    azimuth = azimuth / 360

    # Elevation is in range [-90, 90] degrees
    elevation = elevation / 180 + 0.5

    # Make NumpySample
    return {
        "solar_azimuth": azimuth,
        "solar_elevation": elevation,
    }
