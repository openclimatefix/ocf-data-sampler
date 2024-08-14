"""fill time periods"""

import pandas as pd
import numpy as np


def fill_time_periods(time_periods: pd.DataFrame, freq: pd.Timedelta):
    start_dts = pd.to_datetime(time_periods["start_dt"].values).ceil(freq)
    end_dts = pd.to_datetime(time_periods["end_dt"].values)
    date_ranges = [pd.date_range(start_dt, end_dt, freq=freq) for start_dt, end_dt in zip(start_dts, end_dts)]
    return pd.DatetimeIndex(np.concatenate(date_ranges))
