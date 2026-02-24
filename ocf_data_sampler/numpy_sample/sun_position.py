"""Module for calculating solar position."""

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.time_utils import get_day_fraction, get_day_of_year, get_year


def calculate_azimuth_and_elevation(
    datetimes: NDArray[np.datetime64],
    longitude: float,
    latitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the solar azimuth and elevation using optimised ephemeris function.

    This function was copied and adapted pvlib's `ephemeris()` function [1] with changes to speed
    up the computation of the solar azimuth and elevation. See original function for more details.

    [1] https://github.com/pvlib/pvlib-python/blob/main/pvlib/solarposition.py

    Args:
        datetimes: Datetimes for which to calculate the solar coordinates.
        longitude: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        latitude: Latitude in decimal degrees. Positive north of equator, negative to south.

    Returns:
        np.ndarray: The azimuth of the datetimes in degrees
        np.ndarray: The elevation of the datetimes in degrees
    """
    abber = 20 / 3600.
    LatR = np.radians(latitude)

    # the SPA algorithm needs time to be expressed in terms of
    # decimal UTC hours of the day of the year.
    year_int_arr = get_year(datetimes)

    day_frac = get_day_fraction(datetimes)
    UnivDate = get_day_of_year(datetimes)
    Yr = year_int_arr - 1900
    YrBegin = 365 * Yr + np.floor((Yr - 1) / 4.) - 0.5

    Ezero = YrBegin + UnivDate
    T = Ezero / 36525.

    # Calculate Greenwich Mean Sidereal Time (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    GMSTi = np.mod(GMST0 + 360 * (1.0027379093 * day_frac), 360)

    # Local apparent sidereal time
    LocAST = np.mod((360 + GMSTi + longitude), 360)

    EpochDate = Ezero + day_frac
    T1 = EpochDate / 36525.

    ObliquityR = np.radians(
        23.452294 - 0.0130125 * T1 - 1.64e-06 * T1 ** 2 + 5.03e-07 * T1 ** 3)
    MlPerigee = 281.22083 + 4.70684e-05 * EpochDate + 0.000453 * T1 ** 2 + (
        3e-06 * T1 ** 3)
    MeanAnom = np.mod((358.47583 + 0.985600267 * EpochDate - 0.00015 *
                       T1 ** 2 - 3e-06 * T1 ** 3), 360)
    Eccen = 0.01675104 - 4.18e-05 * T1 - 1.26e-07 * T1 ** 2
    EccenAnom = MeanAnom
    E = 0

    while np.max(abs(EccenAnom - E)) > 0.0001:
        E = EccenAnom
        EccenAnom = MeanAnom + np.degrees(Eccen)*np.sin(np.radians(E))

    TrueAnom = (
        2 * np.mod(np.degrees(np.arctan2(((1 + Eccen) / (1 - Eccen)) ** 0.5 *
                   np.tan(np.radians(EccenAnom) / 2.), 1)), 360))
    EcLon = np.mod(MlPerigee + TrueAnom, 360) - abber
    EcLonR = np.radians(EcLon)
    DecR = np.arcsin(np.sin(ObliquityR)*np.sin(EcLonR))

    RtAscen = np.degrees(np.arctan2(np.cos(ObliquityR)*np.sin(EcLonR),
                                    np.cos(EcLonR)))

    HrAngle = LocAST - RtAscen
    HrAngleR = np.radians(HrAngle)

    SunAz = np.degrees(np.arctan2(-np.sin(HrAngleR),
                                  np.cos(LatR)*np.tan(DecR) -
                                  np.sin(LatR)*np.cos(HrAngleR)))
    SunAz[SunAz < 0] += 360

    SunEl = np.degrees(np.arcsin(
        np.cos(LatR) * np.cos(DecR) * np.cos(HrAngleR) +
        np.sin(LatR) * np.sin(DecR)))

    return SunAz, SunEl


def make_sun_position_numpy_sample(
    datetimes: NDArray[np.datetime64],
    lon: float,
    lat: float,
) -> NumpySample:
    """Creates NumpySample with standardized solar coordinates.

    Args:
        datetimes: Datetimes for which to calculate the solar coordinates.
        lon: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        lat: Latitude in decimal degrees. Positive north of equator, negative to south.
    """
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Normalise
    # Azimuth is in range [0, 360] degrees
    azimuth = azimuth / 360

    # Elevation is in range [-90, 90] degrees
    elevation = elevation / 180 + 0.5

    return {
        "solar_azimuth": azimuth,
        "solar_elevation": elevation,
    }

