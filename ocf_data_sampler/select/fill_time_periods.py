"""Fill time periods between start and end dates at specified frequency"""

import numpy as np
import pandas as pd


def fill_time_periods(time_periods: pd.DataFrame, freq: pd.Timedelta) -> pd.DatetimeIndex:
    """Generate DatetimeIndex for all timestamps between start and end dates"""
    
    start_dts = pd.to_datetime(time_periods["start_dt"].values).ceil(freq)
    end_dts = pd.to_datetime(time_periods["end_dt"].values)
    date_ranges = [
        pd.date_range(start_dt, end_dt, freq=freq)
        for start_dt, end_dt in zip(start_dts, end_dts, strict=False)
    ]
    return pd.DatetimeIndex(np.concatenate(date_ranges))
