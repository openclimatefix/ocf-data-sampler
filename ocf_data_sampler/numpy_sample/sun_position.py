"""Module for calculating solar position."""

import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def ephemeris(time, latitude, longitude, pressure=101325.0, temperature=12.0):
    """
    Python-native solar position calculator.
    The accuracy of this code is not guaranteed.
    Consider using the built-in spa_c code or the PyEphem library.

    Parameters
    ----------
    time : np.datetime64
    latitude : float
        Latitude in decimal degrees. Positive north of equator, negative
        to south.
    longitude : float
        Longitude in decimal degrees. Positive east of prime meridian,
        negative to west.
    pressure : float or Series, default 101325.0
        Ambient pressure (Pascals)
    temperature : float or Series, default 12.0
        Ambient temperature (C)

    """

    abber = 20 / 3600.
    LatR = np.radians(latitude)

    # the SPA algorithm needs time to be expressed in terms of
    # decimal UTC hours of the day of the year.
    date_arr = time.astype("datetime64[D]")
    year_arr = time.astype("datetime64[Y]")
    year_int_arr = time.astype("datetime64[Y]").astype(int) + 1970

    UnivHr = (time - date_arr).astype(int) / (3600*1e9)
    UnivDate = (date_arr - year_arr).astype("int64") + 1
    Yr = year_int_arr - 1900
    YrBegin = 365 * Yr + np.floor((Yr - 1) / 4.) - 0.5

    Ezero = YrBegin + UnivDate
    T = Ezero / 36525.

    # Calculate Greenwich Mean Sidereal Time (GMST)
    GMST0 = 6 / 24. + 38 / 1440. + (
        45.836 + 8640184.542 * T + 0.0929 * T ** 2) / 86400.
    GMST0 = 360 * (GMST0 - np.floor(GMST0))
    GMSTi = np.mod(GMST0 + 360 * (1.0027379093 * UnivHr / 24.), 360)

    # Local apparent sidereal time
    LocAST = np.mod((360 + GMSTi + longitude), 360)

    EpochDate = Ezero + UnivHr / 24.
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
    HrAngle = HrAngle - (360 * (abs(HrAngle) > 180))

    SunAz = np.degrees(np.arctan2(-np.sin(HrAngleR),
                                  np.cos(LatR)*np.tan(DecR) -
                                  np.sin(LatR)*np.cos(HrAngleR)))
    SunAz[SunAz < 0] += 360

    SunEl = np.degrees(np.arcsin(
        np.cos(LatR) * np.cos(DecR) * np.cos(HrAngleR) +
        np.sin(LatR) * np.sin(DecR)))


    # Calculate refraction correction
    Elevation = SunEl
    TanEl = np.tan(np.radians(Elevation))
    Refract = np.zeros(len(time))

    mask = (Elevation > 5) & (Elevation <= 85)
    TanEl_m = TanEl[mask]
    Refract[mask] = (58.1/TanEl_m - 0.07/(TanEl_m**3) + 8.6e-05/(TanEl_m**5))

    mask = (Elevation > -0.575) & (Elevation <= 5)
    Elevation_m = Elevation[mask]
    Refract[mask] = (
        Elevation_m *
        (-518.2 + Elevation_m*(103.4 + Elevation_m*(-12.79 + Elevation_m*0.711))) +
        1735)

    mask = (Elevation > -1) & (Elevation <= -0.575)
    Refract[mask] = -20.774 / TanEl[mask]

    Refract *= (283/(273. + temperature)) * (pressure/101325.) / 3600.

    return {
            'elevation': SunEl,
            'azimuth': SunAz,
        }


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
    solpos = ephemeris(
        time=datetimes,
        longitude=lon,
        latitude=lat,
    )

    return solpos["azimuth"], solpos["elevation"]


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
