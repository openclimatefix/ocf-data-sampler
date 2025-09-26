import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_sample.datetime_features import encode_datetimes


def test_encode_datetimes():
    # Pick summer solstice day and calculate encoding features
    datetimes = pd.to_datetime(["2024-06-20 12:00", "2024-06-20 12:30", "2024-06-20 13:00"])
    features = encode_datetimes(datetimes)

    assert len(features) == 4
    assert all(len(arr) == len(datetimes) for arr in features.values())
    assert (features["date_cos"] != features["date_sin"]).all()

    # Values should be between -1 and 1
    for key in ("date_sin", "date_cos", "time_sin", "time_cos"):
        assert np.all(np.abs(features[key]) <= 1)
