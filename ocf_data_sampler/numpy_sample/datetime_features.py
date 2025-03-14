"""Functions to create trigonometric date and time inputs."""

import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def _get_date_time_in_pi(dt: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Create positional embeddings for the datetimes in radians.

    Args:
        dt: DatetimeIndex to create radian embeddings for

    Returns:
        Tuple of numpy arrays containing radian coordinates for date and time
    """
    day_of_year = dt.dayofyear
    minute_of_day = dt.minute + dt.hour * 60

    time_in_pi = (2 * np.pi) * (minute_of_day / (24 * 60))
    date_in_pi = (2 * np.pi) * (day_of_year / 365)

    return date_in_pi, time_in_pi


def make_datetime_numpy_dict(datetimes: pd.DatetimeIndex, key_prefix: str = "wind") -> NumpySample:
    """Creates dictionary of cyclical datetime features - encoded."""
    date_in_pi, time_in_pi = _get_date_time_in_pi(datetimes)

    time_numpy_sample = {}

    time_numpy_sample[key_prefix + "_date_sin"] = np.sin(date_in_pi)
    time_numpy_sample[key_prefix + "_date_cos"] = np.cos(date_in_pi)
    time_numpy_sample[key_prefix + "_time_sin"] = np.sin(time_in_pi)
    time_numpy_sample[key_prefix + "_time_cos"] = np.cos(time_in_pi)

    return time_numpy_sample
