import numpy as np
import pandas as pd
import pvlib
import pytest

from ocf_data_sampler.numpy_sample.sun_position import (
    calculate_azimuth_and_elevation,
    make_sun_position_numpy_sample,
    ephemeris,
)


@pytest.mark.parametrize("lat", [0, 5, 10, 23.5])
def test_calculate_azimuth_and_elevation(lat):

    # Summer solstice day and sun angle calculation
    datetimes = np.array(["2024-06-20 12:00"], dtype="datetime64[ns]")
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon=0, lat=lat)

    assert len(azimuth) == len(datetimes)
    assert len(elevation) == len(datetimes)
    # Elevation should be close to 90 - (23.5 - lat)
    assert np.abs(elevation - (90 - 23.5 + lat)) < 1


def test_calculate_azimuth_and_elevation_random():
    """Test that the function produces the expected range of azimuths and elevations"""

    np.random.seed(0)

    # Pick day of summer solstice
    datetimes = np.array(["2024-06-20 12:00"], dtype="datetime64[ns]")

    # For 100 random locations - calculate azimuth and elevations
    azimuths, elevations = [], []
    for _ in range(100):
        lon = np.random.uniform(low=0, high=360)
        lat = np.random.uniform(low=-90, high=90)
        azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon=lon, lat=lat)
        azimuths.append(azimuth.item())
        elevations.append(elevation.item())

    azimuths, elevations = np.array(azimuths), np.array(elevations)

    # Assert both azimuth range is [0, 360] and elevation range is [-90, 90]
    assert np.all((azimuths >= 0) & (azimuths <= 360))
    assert np.all((elevations >= -90) & (elevations <= 90))
    assert azimuths.min() < 30 and azimuths.max() > 330
    assert elevations.min() < -70 and elevations.max() > 70


def test_make_sun_position_numpy_sample():
    datetimes = pd.date_range("2024-06-20 12:00", "2024-06-20 16:00", freq="30min").values
    sample = make_sun_position_numpy_sample(datetimes, lon=0, lat=51.5)

    # Assertion accounting for solar coord normalisation
    assert {"solar_elevation", "solar_azimuth"} <= set(sample)
    assert np.all((sample["solar_elevation"] >= 0) & (sample["solar_elevation"] <= 1))
    assert np.all((sample["solar_azimuth"] >= 0) & (sample["solar_azimuth"] <= 1))


def test_ephemeris():

    latitude = 55.201231
    longitude = -6.656644

    # Randomly sample datetimes between the two extremes
    t_min = np.datetime64("1900-01-01 00:00", "ns")
    t_max = np.datetime64("2100-01-01 00:00", "ns")
    dt = t_max - t_min
    random_datetimes = t_min + (np.random.uniform(size=1000) * dt)

    # Test the ephemeris algorithm here aginst pvlib
    solar_coords = ephemeris(random_datetimes, longitude=longitude, latitude=latitude)
    pvlib_solar_coords = pvlib.solarposition.get_solarposition(
        time=pd.to_datetime(random_datetimes),
        longitude=longitude,
        latitude=latitude,
        method="ephemeris",
    )

    assert np.isclose(solar_coords["elevation"], pvlib_solar_coords["elevation"].values).all()
    assert np.isclose(solar_coords["azimuth"], pvlib_solar_coords["azimuth"].values).all()

    