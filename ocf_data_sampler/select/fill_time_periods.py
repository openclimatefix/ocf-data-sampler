"""Fill time periods between specified start and end dates."""

import numpy as np
import pandas as pd


def fill_time_periods(time_periods: pd.DataFrame, freq: pd.Timedelta) -> pd.DatetimeIndex:
    """Create range of timestamps between given start and end times.

    Each of the continuous periods (i.e. each row of the input DataFrame) is filled with the
    specified frequency.

    Args:
        time_periods: DataFrame with columns 'start_dt' and 'end_dt'
        freq: Frequency to fill time periods with
    """
    start_dts = pd.to_datetime(time_periods["start_dt"].values).ceil(freq)
    end_dts = pd.to_datetime(time_periods["end_dt"].values)
    date_ranges = [
        pd.date_range(start_dt, end_dt, freq=freq)
        for start_dt, end_dt in zip(start_dts, end_dts, strict=False)
        ]
    return pd.DatetimeIndex(np.concatenate(date_ranges))
