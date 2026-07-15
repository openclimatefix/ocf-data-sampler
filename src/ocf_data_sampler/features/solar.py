"""Solar azimuth and elevation via an optimised port of pvlib's ephemeris algorithm.

This module contains a copy of the `ephemeris()` function from the pvlib-python
project [1], adapted for speed.

Modifications copyright 2026 Open Climate Fix. Licensed under the MIT License.

Modifications from the original include:
- Accepts numpy datetime64 arrays directly (no pandas DatetimeIndex)
- No atmospheric refraction correction (returns true elevation)
- Time components computed via ocf_data_sampler.common.time_utils

References:
    [1] https://github.com/pvlib/pvlib-python/blob/main/pvlib/solarposition.py
"""
# The following notice is retained from pvlib-python as required by its license:
#
# BSD 3-Clause License
#
# Copyright (c) 2023 pvlib python Contributors
# Copyright (c) 2014 PVLIB python Development Team
# Copyright (c) 2013 Sandia National Laboratories
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
#   Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.time_utils import get_day_fraction, get_day_of_year, get_year


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
