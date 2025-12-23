"""Functions to create trigonometric date and time inputs."""

from typing import Literal

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


def get_t0_embedding(
    t0: pd.Timestamp,
    periods: list[str],
    embeddings: list[Literal["cyclic", "linear"]],
) -> dict[str, np.ndarray]:
    """Creates dictionary of sin and cos t0 time embeddings.

    Args:
        t0: The time to create sin-cos embeddings for
        periods: List of periods to encode (e.g., "1h", "Nh", "1y", "Ny")
        embeddings: How to represent each of these periods. Either "cyclic" or "linear". When cyclic
            the period is sin-cos embedded, else it is 0-1 scaled as fraction through the period.
    """
    features = []

    if len(periods)!=len(embeddings):
        raise ValueError("`periods` and `embeddings` must be the same length")

    for period_str, embedding in zip(periods, embeddings, strict=True):

        if period_str.endswith("h"):
            period_hours = int(period_str.removesuffix("h"))
            if not (1<=period_hours<=24):
                raise ValueError("The period in hours must be in interval [1, 24]")
            frac = (t0.hour + t0.minute / 60) / period_hours

        elif period_str.endswith("y"):
            period_years = int(period_str.removesuffix("y"))
            if not period_years > 0:
                raise ValueError("The period in years must be >0")
            days_in_year = 366 if t0.is_leap_year else 365
            frac = (((t0.dayofyear-1) / days_in_year) + t0.year % period_years) / period_years

        else:
            raise ValueError(f"Invalid period format: {period_str}")

        if embedding=="cyclic":
            radians = 2 * np.pi * frac
            features.extend([np.sin(radians), np.cos(radians)])
        elif embedding=="linear":
            features.append(frac)
        else:
            raise ValueError(f"embedding option {embedding} not recognised")

    return {"t0_embedding": np.array(features, dtype=np.float32)}
