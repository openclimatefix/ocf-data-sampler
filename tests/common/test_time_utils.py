import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.common.time_utils import (
    date_range,
    datetime_ceil,
    datetime_floor,
    get_day_fraction,
    get_day_of_year,
    get_hour,
    get_minute,
    get_posix_timestamp,
    get_year,
    is_leap_year,
)


@pytest.fixture()
def example_datetimes():
    t_min = np.datetime64("1900-01-01 00:00", "ns")
    t_max = np.datetime64("2100-01-01 00:00", "ns")
    dt = t_max - t_min
    # Random times almost never land on or just beside a rounding boundary, where floor and ceil
    # are easiest to get wrong, so add times on and 1 ns either side of one
    boundary = np.datetime64("2000-01-01T12:00", "ns")
    one_ns = np.timedelta64(1, "ns")
    boundary_datetimes = [boundary - one_ns, boundary, boundary + one_ns]
    return np.concatenate([t_min + (np.random.uniform(size=1000) * dt), boundary_datetimes])


def test_get_year(example_datetimes):
    assert np.array_equal(
        get_year(example_datetimes),
        pd.to_datetime(example_datetimes).year.values,
    )


def test_get_hour(example_datetimes):
    assert np.array_equal(
        get_hour(example_datetimes),
        pd.to_datetime(example_datetimes).hour.values,
    )


def test_get_minute(example_datetimes):
    assert np.array_equal(
        get_minute(example_datetimes),
        pd.to_datetime(example_datetimes).minute.values,
    )


def test_get_day_of_year(example_datetimes):
    assert np.array_equal(
        get_day_of_year(example_datetimes),
        pd.to_datetime(example_datetimes).dayofyear.values,
    )


def test_get_day_fraction(example_datetimes):
    expected = (
        (example_datetimes - example_datetimes.astype("datetime64[D]")) / np.timedelta64(1, "D")
    ).astype(np.float64)
    assert np.allclose(get_day_fraction(example_datetimes), expected)


def test_is_leap_year(example_datetimes):
    assert np.array_equal(
        is_leap_year(example_datetimes),
        pd.to_datetime(example_datetimes).is_leap_year,
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
def test_datetime_ceil(example_datetimes, freq):
    assert np.array_equal(
        datetime_ceil(example_datetimes, freq=freq),
        pd.to_datetime(example_datetimes).ceil(pd.Timedelta(freq)).values,
    )


@pytest.mark.parametrize(
    "freq",
    [np.timedelta64(5, "m"), np.timedelta64(30, "m"), np.timedelta64(1, "h")],
)
def test_datetime_floor(example_datetimes, freq):
    assert np.array_equal(
        datetime_floor(example_datetimes, freq=freq),
        pd.to_datetime(example_datetimes).floor(pd.Timedelta(freq)).values,
    )


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_get_posix_timestamp(unit):
    datetimes = np.array(["2024-01-01T00:00", "2024-01-01T01:00"], dtype=f"datetime64[{unit}]")
    timestamps = get_posix_timestamp(datetimes)
    assert timestamps.dtype == np.int64
    np.testing.assert_array_equal(timestamps, [1704067200, 1704070800])
    assert get_posix_timestamp(datetimes[0]) == 1704067200
