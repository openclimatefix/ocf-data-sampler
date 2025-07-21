"""Functions to create trigonometric date and time inputs."""

import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def encode_datetimes(datetimes: pd.DatetimeIndex) -> NumpySample:
    """Creates dictionary of sin and cos datetime embeddings.

    Args:
        datetimes: DatetimeIndex to create radian embeddings for

    Returns:
        Dictionary of datetime encodings
    """
    day_of_year = datetimes.dayofyear
    minute_of_day = datetimes.minute + datetimes.hour * 60

    time_in_radians = (2 * np.pi) * (minute_of_day / (24 * 60))
    date_in_radians = (2 * np.pi) * (day_of_year / 365)

    return {
        "date_sin": np.sin(date_in_radians),
        "date_cos": np.cos(date_in_radians),
        "time_sin": np.sin(time_in_radians),
        "time_cos": np.cos(time_in_radians),
    }
