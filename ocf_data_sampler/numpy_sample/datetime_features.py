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


def get_t0_sin_cos_embedding(t0: pd.Timestamp, periods: list[str]) -> dict[str, np.ndarray]:
    """Creates dictionary of sin and cos t0 time embeddings.

    Args:
        t0: The time to create sin-cos embeddings for
        periods: List of periods to cos-sin encode (e.g., "1h", "Nh", "1y", "Ny")
    """

    period_fracs = []
    for period_str in periods:

        if period_str.endswith("h"):
            period_hours = int(period_str.removesuffix("h"))
            if not period_hours>0:
                raise ValueError("The period in hours must be >0")
            frac = (t0.hour + t0.minute / 60) / period_hours

        elif period_str.endswith("y"):
            period_years = int(period_str.removesuffix("y"))
            if not period_years > 0:
                raise ValueError("The period in years must be >0")
            days_in_year = 366 if t0.is_leap_year else 365
            frac = (((t0.dayofyear-1) / days_in_year) + t0.year % period_years) / period_years
        
        else:
            raise ValueError(f"Invalid period format: {period_str}")
        
        period_fracs.append(frac)

    period_radians = 2 * np.pi * np.array(period_fracs, dtype=np.float32)

    # Interleave Sin/Cos (using the slicing method)
    t0_embedding = np.empty(len(periods)*2, dtype=np.float32)
    t0_embedding[::2] = np.sin(period_radians)
    t0_embedding[1::2] = np.cos(period_radians)
    
    return {"t0_embedding": t0_embedding}
