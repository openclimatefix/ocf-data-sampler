"""Select time periods"""

import pandas as pd
import numpy as np



def fill_time_periods(time_periods: pd.DataFrame, freq: pd.Timedelta):
    datetimes = []
    for _, row in time_periods.iterrows():
        start_dt = pd.Timestamp(row["start_dt"]).ceil(freq)
        end_dt = pd.Timestamp(row["end_dt"])
        datetimes.append(pd.date_range(start_dt, end_dt, freq=freq))

    return pd.DatetimeIndex(np.concatenate(datetimes))
