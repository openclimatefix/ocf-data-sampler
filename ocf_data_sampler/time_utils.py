"""Module for datetime utilities."""
import numpy as np
from numpy.typing import NDArray


def minutes(minutes: int | list[int]) -> np.timedelta64 | NDArray[np.timedelta64]:
    """Timedelta minutes.

    Args:
        minutes: the number of minutes, single value or list
    """
    if isinstance(minutes, int | np.integer):
        return np.timedelta64(minutes, "m")
    return np.asarray(minutes, dtype="timedelta64[m]")


def get_hour(datetimes: np.datetime64 | NDArray[np.datetime64]) -> np.int32 | NDArray[np.int32]:
    """Get the hour of the day.

    Args:
        datetimes: the datetimes to get hour for
    """
    # Convert to hours since epoch, then mod 24 to get hour-of-day
    return datetimes.astype("datetime64[h]").astype(np.int32) % 24


def get_minute(datetimes: np.datetime64 | NDArray[np.datetime64]) -> np.int32 | NDArray[np.int32]:
    """Get the minute past the hour.

    Args:
        datetimes: the datetimes to get day of year for
    """
    # Convert to minutes since epoch, then mod 60 for minute-of-hour
    return datetimes.astype("datetime64[m]").astype(np.int32) % 60


def get_year(datetimes: np.datetime64 | NDArray[np.datetime64]) -> np.int32 | NDArray[np.int32]:
    """Get the year.

    Args:
        datetimes: the datetimes to get year for
    """
    # Convert to years since epoch, then add 1970 for year
    return datetimes.astype("datetime64[Y]").astype(np.int32) + 1970


def get_day_of_year(
    datetimes: np.datetime64 | NDArray[np.datetime64],
) -> np.int32 | NDArray[np.int32]:
    """Get the integer day of year i.e. 1-365/366 depending on leap year.

    Args:
        datetimes: the datetimes to get day of year for
    """
    day_arr = datetimes.astype("datetime64[D]")
    year_start_arr = day_arr.astype("datetime64[Y]").astype("datetime64[D]")
    return (day_arr - year_start_arr).astype(np.int32) + 1


def get_day_fraction(
    datetimes: np.datetime64 | NDArray[np.datetime64],
) -> np.float64 | NDArray[np.float64]:
    """Get the fraction through the day.

    Args:
        datetimes: the datetimes to get day fraction for
    """
    day_start = datetimes.astype("datetime64[D]")
    return ((datetimes - day_start) / np.timedelta64(1, "D")).astype(np.float64)


def get_is_leap_year(
    datetimes: np.datetime64 | NDArray[np.datetime64],
) ->  np.bool_ | NDArray[np.bool_]:
    """Get whether the datetime is in a leap year.

    Args:
        datetimes: the datetimes to get leap year for
    """
    years = get_year(datetimes)
    return (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))


def date_range(
    start: np.datetime64,
    end: np.datetime64,
    freq: np.timedelta64,
) -> NDArray[np.datetime64]:
    """Create a date range inclusive of the end date with the given step size.

    Args:
        start: the start date
        end: the end date
        freq: the frequency to step by
    """
    # Make end inclusive by entending by the smallest possible amount (1 nanosecond)
    return np.arange(start, end + np.timedelta64(1, "ns"), freq)


def _floor_ceil_check_freq(freq: np.timedelta64) -> None:
    """Check that the frequency is a factor or multiple of 1 hour and a factor of 1 day.

    This is required for the floor and ceil functions to work correctly.
    """
    # For simplicity don't support frequencies that aren't factors or multiples of 1 hour.
    # For example what would it mean to ceil/floor to 45 minutes?
    if (np.timedelta64(1, "h") / freq)%1 != 0 and (freq/np.timedelta64(1, "h") )%1 != 0:
        raise ValueError(f"Frequency {freq} must be factor of 1 hour or multiple of hour.")
    # Also don't support frequecies that aren't factors of 1 day.
    # For example what would it mean to ceil/floor to 7 hours?
    if (np.timedelta64(1, "D") / freq)%1 !=0:
        raise ValueError(f"Frequency {freq} must be factor of 1 day.")


def datetime_ceil(
    datetimes: np.datetime64 | NDArray[np.datetime64],
    freq: np.timedelta64,
) -> np.datetime64 | NDArray[np.datetime64]:
    """Ceil a datetime to the nearest frequency.

    Args:
        datetimes: the datetimes to ceil
        freq: the frequency to ceil to
    """
    _floor_ceil_check_freq(freq)
    epoch_datetime = np.datetime64("1970-01-01", "ns")
    periods_since_epoch = np.ceil((datetimes - epoch_datetime) / freq)
    return ((periods_since_epoch * freq) + epoch_datetime).astype(datetimes.dtype)


def datetime_floor(
    datetimes: np.datetime64 | NDArray[np.datetime64],
    freq: np.timedelta64,
) -> np.datetime64 | NDArray[np.datetime64]:
    """Floor a datetime to the nearest frequency.

    Args:
        datetimes: the datetimes to floor
        freq: the frequency to floor to
    """
    _floor_ceil_check_freq(freq)
    epoch_datetime = np.datetime64("1970-01-01", "ns")
    periods_since_epoch = np.floor((datetimes - epoch_datetime) / freq)
    return ((periods_since_epoch * freq) + epoch_datetime).astype(datetimes.dtype)


def get_posix_timestamp(
    datetimes: np.datetime64 | NDArray[np.datetime64],
) -> np.float64 | NDArray[np.float64]:
    """Get the POSIX timestamp (seconds since 1970-01-01) for the given datetimes.

    Args:
        datetimes: the datetimes to get POSIX timestamps for
    """
    return datetimes.astype("datetime64[s]").astype(np.float64)
