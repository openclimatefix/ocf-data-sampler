"""Functions to create trigonometric date and time inputs."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.time_utils import (
    get_day_fraction,
    get_day_of_year,
    get_hour,
    get_is_leap_year,
    get_minute,
    get_year,
)


def encode_datetimes(datetimes: NDArray[np.datetime64]) -> NumpySample:
    """Creates dictionary of sin and cos datetime embeddings.

    Args:
        datetimes: datetime array to create radian embeddings for

    Returns:
        Dictionary of datetime encodings
    """
    day_fraction = get_day_fraction(datetimes)
    day_of_year = get_day_of_year(datetimes)

    time_in_radians = (2 * np.pi) * day_fraction
    date_in_radians = (2 * np.pi) * (day_of_year / 365)

    return {
        "date_sin": np.sin(date_in_radians),
        "date_cos": np.cos(date_in_radians),
        "time_sin": np.sin(time_in_radians),
        "time_cos": np.cos(time_in_radians),
    }


def get_t0_embedding(
    t0: np.datetime64,
    embeddings: list[tuple[str, Literal["cyclic", "linear"]]],
) -> dict[str, np.ndarray]:
    """Creates dictionary of t0 time embeddings.

    Args:
        t0: The time to create sin-cos embeddings for
        embeddings: The periods to encode (e.g., "1h", "Nh", "1y", "Ny") and their representation
            (either "cyclic" or "linear"). When cyclic, the period is sin-cos embedded, else it is
            0-1 scaled as fraction through the period. Note that using "cyclic" adds 2 elements to
            the output vector to embed a period whilst "linear" adds only 1 element.
    """
    features = []

    for period_str, embedding_type in embeddings:

        if period_str.endswith("h"):
            period_hours = int(period_str.removesuffix("h"))
            frac = (get_hour(t0) + get_minute(t0) / 60) / period_hours

        elif period_str.endswith("y"):
            period_years = int(period_str.removesuffix("y"))
            days_in_year = 366 if get_is_leap_year(t0) else 365
            frac = (
                (((get_day_of_year(t0)-1) / days_in_year) + get_year(t0) % period_years)
                / period_years
            )

        if embedding_type=="cyclic":
            radians = 2 * np.pi * frac
            features.extend([np.sin(radians), np.cos(radians)])

        elif embedding_type=="linear":
            features.append(frac)

    return {"t0_embedding": np.array(features, dtype=np.float32)}
