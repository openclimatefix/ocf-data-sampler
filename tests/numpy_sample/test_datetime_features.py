import numpy as np
import pandas as pd

from ocf_data_sampler.numpy_sample.datetime_features import (
    encode_datetimes,
    get_t0_sin_cos_embedding,
)


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


def test_get_t0_sin_cos_embedding():

    def check(t0s, period_strs, xs, period_floats):
        # Test the results are expected for each t0 time
        for x, t0 in zip(xs, t0s):
            results = get_t0_sin_cos_embedding(t0, periods=period_strs)["t0_embedding"]

            expected_results = []
            for p in period_floats:
                expected_results.extend([np.sin(2*np.pi*(x / p)), np.cos(2*np.pi*(x / p))])
            expected_results = np.array(expected_results)
            
            if not np.allclose(results, expected_results, atol=1e-6):
                raise ValueError(f"{results}!={expected_results}")

    # Define some t0 times and periods to check
    t0s = pd.date_range("2024-01-01 00:00", "2024-01-01 06:00")
    period_strs = ["1h", "2h", "6h"]

    # Equivalent times and periods in float form
    xs = np.linspace(0, 6, num=len(t0s))
    period_floats = [1,2,6]

    check(t0s, period_strs, xs, period_floats)

    # Repeat the check focusing on year periods rather than hours
    t0s = pd.to_datetime(
        [
            "2020-01-01 00:00", "2020-01-01 23:30", "2020-01-02 00:00",
            "2020-06-10 00:00", "2021-01-01 00:00", "2021-01-02 00:00",
        ]
    )
    period_strs = ["1y", "2y"]

    # Equivalent times and periods in float form
    # Note:
    # - When doing year encoding we don't consider time of day
    # - 2020 is a leap year but 2021 is not
    # - 2020-06-10 is the 162nd day of that year
    xs = np.array([0, 0, 1/366, 161/366, 1, 1+1/365], dtype=np.float32)
    period_floats = [1, 2]

    check(t0s, period_strs, xs, period_floats)

