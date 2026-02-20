"""Fill time periods between specified start and end dates."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ocf_data_sampler.time_utils import date_range, datetime_ceil


def fill_time_periods(time_periods: pd.DataFrame, freq: np.timedelta64) -> NDArray[np.datetime64]:
    """Create range of timestamps between given start and end times.

    Each of the continuous periods (i.e. each row of the input DataFrame) is filled with the
    specified frequency.

    Args:
        time_periods: DataFrame with columns 'start_dt' and 'end_dt'
        freq: Frequency to fill time periods with
    """
    start_dts = datetime_ceil(time_periods["start_dt"].values, freq)
    end_dts = time_periods["end_dt"].values
    date_ranges = [
        date_range(start_dt, end_dt, freq=freq)
        for start_dt, end_dt in zip(start_dts, end_dts, strict=False)
    ]
    return np.concatenate(date_ranges)
