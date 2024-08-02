"""Select time periods"""

import pandas as pd
import  numpy as np
from datetime import timedelta



def fill_time_periods(time_periods: pd.DataFrame, freq: timedelta):
    datetimes = []
    for _, row in time_periods.iterrows():
        start_dt = row["start_dt"]
        end_dt = row["end_dt"]
        datetimes.append(pd.date_range(start_dt, end_dt, freq=freq))

    return pd.DatetimeIndex(np.concatenate(datetimes))
