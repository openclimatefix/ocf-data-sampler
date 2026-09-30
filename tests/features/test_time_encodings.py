import numpy as np
import pandas as pd

from ocf_data_sampler.features.time_encodings import (
    encode_datetimes,
    encode_t0,
)


def test_encode_datetimes():
    # Pick summer solstice day and calculate encoding features
    datetimes = pd.to_datetime(["2024-06-20 12:00", "2024-06-20 12:30", "2024-06-20 13:00"]).values
    features = encode_datetimes(datetimes)

    assert len(features) == 4
    assert all(len(arr) == len(datetimes) for arr in features.values())
    assert (features["date_cos"] != features["date_sin"]).all()

    # Values should be between -1 and 1
    for key in ("date_sin", "date_cos", "time_sin", "time_cos"):
        assert np.all(np.abs(features[key]) <= 1)
        assert features[key].dtype == np.float32

    # The date encoding must agree with encode_t0 and must not alias across the year boundary
    for datetime in (np.datetime64("2023-01-01"), np.datetime64("2024-12-31")):
        date_sin = encode_datetimes(np.array([datetime]))["date_sin"][0]
        assert date_sin == encode_t0(datetime, [("1y", "cyclic")])[0]

    assert (
        encode_datetimes(np.array([np.datetime64("2023-01-01")]))["date_sin"][0]
        != encode_datetimes(np.array([np.datetime64("2024-12-31")]))["date_sin"][0]
    )


def test_encode_t0():

    # Check hourly embedding codepath

    # Define some t0 times to check for
    t0s = pd.date_range("2024-01-01 00:00", "2024-01-01 23:55", freq="5min").values
    # These are the fractional hour-of-day for the above
    hour_floats = np.arange(0, 24, 5 / 60)

    # Check over multiple frequencies
    for h_freq in [1, 2, 3]:
        hours = f"{h_freq}h"
        linear_embeddings = [(hours, "linear")]
        cyclic_embeddings = [(hours, "cyclic")]

        for t0, hour_float in zip(t0s, hour_floats, strict=True):
            # Check linear encodings
            expected_linear = (hour_float % h_freq) / h_freq
            result = encode_t0(t0, linear_embeddings)
            # Linear embedding should be in range [0, 1]
            assert (result >= 0) & (result <= 1)
            assert np.isclose(expected_linear, result)

            # Check cyclic encodings
            radians = 2 * np.pi * expected_linear
            expected_cyclic = [np.sin(radians), np.cos(radians)]
            result = encode_t0(t0, cyclic_embeddings)
            # Cyclic embedding should be in range [-1, 1]
            assert ((result >= -1) & (result <= 1)).all()
            assert np.allclose(expected_cyclic, result)

            # Check multi-embedding ordering

            # [linear, cyclic]
            all_embeddings = linear_embeddings + cyclic_embeddings
            result = encode_t0(t0, all_embeddings)
            expected = [expected_linear, *expected_cyclic]
            assert np.allclose(expected, result)

            # [cyclic, linear]
            all_embeddings = cyclic_embeddings + linear_embeddings
            result = encode_t0(t0, all_embeddings)
            expected = [*expected_cyclic, expected_linear]
            assert np.allclose(expected, result)

    # Check remaining portion of yearly embedding codepath for full coverage

    t0_year_floats = [
        ("2020-01-01 00:00", 2020),
        ("2020-01-01 23:30", 2020),  # Time of day is ignored
        ("2020-01-02 00:00", 2020 + 1 / 366),  # 2020 is a leap year
        ("2020-06-10 00:00", 2020 + 161 / 366),  # 2020-06-10 is the 162nd day of that year
        ("2021-01-01 00:00", 2021),
        ("2021-01-02 00:00", 2021 + 1 / 365),  # 2020 is not a leap year
    ]

    for y_freq in [1, 2]:
        # We only need to test linear embeddding since the cyclic part is tested for hourly and
        # these share the same codepath
        linear_embeddings = [(f"{y_freq}y", "linear")]
        for t0_str, year_float in t0_year_floats:
            # Check linear encodings
            expected_linear = (year_float % y_freq) / y_freq
            result = encode_t0(np.datetime64(t0_str), linear_embeddings)
            assert (result >= 0) & (result <= 1)
            assert np.isclose(expected_linear, result)
