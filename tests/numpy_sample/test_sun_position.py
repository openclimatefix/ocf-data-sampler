import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.numpy_sample.sun_position import (
    calculate_azimuth_and_elevation,
    make_sun_position_numpy_sample,
)


@pytest.mark.parametrize("lat", [0, 5, 10, 23.5])
def test_calculate_azimuth_and_elevation(lat):
    # Pick the day of the summer solstice
    datetimes = pd.to_datetime(["2024-06-20 12:00"])

    # Calculate sun angles
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon=0, lat=lat)

    assert len(azimuth) == len(datetimes)
    assert len(elevation) == len(datetimes)

    # elevation should be close to (90 - (23.5-lat) degrees
    assert np.abs(elevation - (90 - 23.5 + lat)) < 1


def test_calculate_azimuth_and_elevation_random():
    """Test that the function produces the expected range of azimuths and elevations"""

    # Set seed so we know the test should pass
    np.random.seed(0)

    # Pick the day of the summer solstice
    datetimes = pd.to_datetime(["2024-06-20 12:00"])

    # Pick 100 random locations and measure their azimuth and elevations
    azimuths = []
    elevations = []

    for _ in range(100):
        lon = np.random.uniform(low=0, high=360)
        lat = np.random.uniform(low=-90, high=90)

        # Calculate sun angles
        azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon=lon, lat=lat)

        azimuths.append(azimuth.item())
        elevations.append(elevation.item())

    azimuths = np.array(azimuths)
    elevations = np.array(elevations)

    assert (azimuths >= 0).all() and (azimuths <= 360).all()
    assert (elevations >= -90).all() and (elevations <= 90).all()

    # Azimuth range is [0, 360]
    assert azimuths.min() < 30
    assert azimuths.max() > 330

    # Elevation range is [-90, 90]
    assert elevations.min() < -70
    assert elevations.max() > 70


def test_make_sun_position_numpy_sample():
    datetimes = pd.date_range("2024-06-20 12:00", "2024-06-20 16:00", freq="30min")
    lon, lat = 0, 51.5

    sample = make_sun_position_numpy_sample(datetimes, lon, lat)

    assert "solar_elevation" in sample
    assert "solar_azimuth" in sample

    # The solar coords are normalised in the function
    assert (sample["solar_elevation"] >= 0).all()
    assert (sample["solar_elevation"] <= 1).all()
    assert (sample["solar_azimuth"] >= 0).all()
    assert (sample["solar_azimuth"] <= 1).all()
