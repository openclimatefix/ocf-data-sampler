import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_batch.datetime_features import make_datetime_numpy_batch

from ocf_data_sampler.numpy_batch import GSPBatchKey


def test_calculate_azimuth_and_elevation():

    # Pick the day of the summer solstice
    datetimes = pd.to_datetime(["2024-06-20 12:00", "2024-06-20 12:30", "2024-06-20 13:00"])

    # Calculate sun angles
    datetime_features = make_datetime_numpy_batch(datetimes)

    assert len(datetime_features) == 4

    assert len(datetime_features["wind_date_sin"]) == len(datetimes)
    assert (datetime_features["wind_date_cos"] != datetime_features["wind_date_sin"]).all()

    # assert all values are between -1 and 1
    assert all(np.abs(datetime_features["wind_date_sin"]) <= 1)
    assert all(np.abs(datetime_features["wind_date_cos"]) <= 1)
    assert all(np.abs(datetime_features["wind_time_sin"]) <= 1)
    assert all(np.abs(datetime_features["wind_time_cos"]) <= 1)
