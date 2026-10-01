import itertools

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

    # Check some input-output pairs for both linear and cyclic encodings for hourly frequencies
    # These are [datetime, frequency, expected linear fraction through the period] pairs
    input_output_pairs = [
        # 1-hour embeddings
        ("2020-01-01 00:00", "1h", 0),
        ("2020-01-01 06:10", "1h", 1 / 6),
        ("2020-01-01 06:30", "1h", 0.5),
        ("2020-01-01 06:55", "1h", 55 / 60),
        # N-hour embeddings
        ("2020-01-01 08:00", "2h", 0),
        ("2020-01-01 09:00", "2h", 0.5),
        ("2020-01-01 06:55", "2h", 55 / 120),
        ("2020-01-01 08:00", "3h", 2 / 3),
        ("2020-01-01 13:10", "5h", (3 + 1 / 6) / 5),
        ("2020-01-01 08:00", "24h", 8 / 24),
        ("2020-01-01 23:30", "24h", 23.5 / 24),
        # 1-year embeddings
        ("2020-01-01 00:00", "1y", 0),
        ("2021-01-01 00:00", "1y", 0),
        ("2020-01-01 23:30", "1y", 0),  # Time of day is ignored
        ("2020-01-02 00:00", "1y", 1 / 366),  # 2020 is a leap year hence 366 days
        ("2020-06-10 00:00", "1y", 161 / 366),  # 2020-06-10 is the 162nd day of that year
        ("2021-06-10 00:00", "1y", 160 / 365),  # 2021-06-10 is the 161st day of that year
        ("2021-01-01 00:00", "1y", 0),
        ("2021-01-02 00:00", "1y", 1 / 365),  # 2020 is not a leap year
        # N-year embeddings (we are unlikely to use these in practice, but they are supported)
        ("2020-01-01 00:00", "2y", 0),
        ("2021-01-01 00:00", "2y", 0.5),
        ("2022-06-10 00:00", "2y", (160 / 365) / 2),  # 2022-06-10 is the 161st day of that year
        ("2021-01-01 00:00", "3y", 2 / 3),  # 2019 is divisible by 3. This is 2 years after
        # 2020 is a leap year but each full year as a single period for N-year embeddings
        # Hence the difference in the similar rows below
        ("2018-01-02 00:00", "2y", (1 / 365) / 2),
        ("2020-01-02 00:00", "2y", (1 / 366) / 2),
    ]

    for t0_str, freq_str, linear_frac in input_output_pairs:
        # Check linear encodings
        linear_result = encode_t0(np.datetime64(t0_str), [(freq_str, "linear")])
        assert (linear_result >= 0) & (linear_result <= 1)
        assert np.isclose(linear_frac, linear_result)

        # Check cyclic encodings
        cyclic_result = encode_t0(np.datetime64(t0_str), [(freq_str, "cyclic")])
        expected_cyclic = [np.sin(2 * np.pi * linear_frac), np.cos(2 * np.pi * linear_frac)]
        assert ((cyclic_result >= -1) & (cyclic_result <= 1)).all()
        assert np.allclose(cyclic_result, expected_cyclic)

    # Check ordering is as expected when configured with multiple embeddings
    t0 = np.datetime64("2020-06-10 06:30")
    embedding_result_options = [
        (("1h", "cyclic"), [np.sin(2 * np.pi * 0.5), np.cos(2 * np.pi * 0.5)]),
        (("1h", "linear"), [0.5]),
        (("2h", "linear"), [0.25]),
        (("1y", "cyclic"), [np.sin(2 * np.pi * (161 / 366)), np.cos(2 * np.pi * (161 / 366))]),
    ]
    # Check ordering for all permutations of 2, 3, or 4 embeddings
    for n in [2, 3, 4]:
        for embedding_result_set in itertools.permutations(embedding_result_options, n):
            embeddings, expected_list = list(zip(*embedding_result_set, strict=False))
            result = encode_t0(t0, embeddings)
            assert np.allclose(result, np.concatenate(expected_list))
