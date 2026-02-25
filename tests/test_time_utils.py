import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.time_utils import (
    date_range,
    datetime_ceil,
    datetime_floor,
    get_day_fraction,
    get_day_of_year,
    get_hour,
    get_is_leap_year,
    get_minute,
    get_posix_timestamp,
    get_year,
)


@pytest.fixture()
def random_datetimes():
    t_min = np.datetime64("1900-01-01 00:00", "ns")
    t_max = np.datetime64("2100-01-01 00:00", "ns")
    dt = t_max - t_min
    return t_min + (np.random.uniform(size=1000) * dt)


def test_get_year(random_datetimes):
    assert np.array_equal(
        get_year(random_datetimes),
        pd.to_datetime(random_datetimes).year.values,
    )


def test_get_hour(random_datetimes):
    assert np.array_equal(
        get_hour(random_datetimes),
        pd.to_datetime(random_datetimes).hour.values,
    )


def test_get_minute(random_datetimes):
    assert np.array_equal(
        get_minute(random_datetimes),
        pd.to_datetime(random_datetimes).minute.values,
    )


def test_get_day_of_year(random_datetimes):
    assert np.array_equal(
        get_day_of_year(random_datetimes),
        pd.to_datetime(random_datetimes).dayofyear.values,
    )


def test_get_day_fraction(random_datetimes):
    expected = (
        (random_datetimes - random_datetimes.astype("datetime64[D]")) / np.timedelta64(1, "D")
    ).astype(np.float64)
    assert np.allclose(get_day_fraction(random_datetimes), expected)


def test_get_is_leap_year(random_datetimes):
    assert np.array_equal(
        get_is_leap_year(random_datetimes),
        pd.to_datetime(random_datetimes).is_leap_year,
    )


def test_date_range():
    start = np.datetime64("2024-01-01")
    end = np.datetime64("2024-01-10")
    freq = np.timedelta64(2, "D")
    assert np.array_equal(
        date_range(start, end, freq),
        pd.date_range(start, end, freq="2D").values,
    )


@pytest.mark.parametrize(
    "freq",
    [np.timedelta64(5, "m"), np.timedelta64(30, "m"), np.timedelta64(1, "h")],
)
def test_datetime_ceil(random_datetimes, freq):
    assert np.array_equal(
        datetime_ceil(random_datetimes, freq=freq),
        pd.to_datetime(random_datetimes).ceil(pd.Timedelta(freq)).values,
    )


@pytest.mark.parametrize(
    "freq",
    [np.timedelta64(5, "m"), np.timedelta64(30, "m"), np.timedelta64(1, "h")],
)
def test_datetime_floor(random_datetimes, freq):
    assert np.array_equal(
        datetime_floor(random_datetimes, freq=freq),
        pd.to_datetime(random_datetimes).floor(pd.Timedelta(freq)).values,
    )


def test_get_posix_timestamp(random_datetimes):
    expected = np.array([t.timestamp() for t in pd.to_datetime(random_datetimes)])
    assert np.isclose(
        get_posix_timestamp(random_datetimes),
        expected,
        rtol=0,
        atol=1e-3,
    ).all()
