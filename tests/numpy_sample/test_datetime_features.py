import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.numpy_sample.datetime_features import make_datetime_numpy_dict


def test_calculate_azimuth_and_elevation():

    # Pick the day of the summer solstice
    datetimes = pd.to_datetime(["2024-06-20 12:00", "2024-06-20 12:30", "2024-06-20 13:00"])

    # Calculate sun angles
    datetime_features = make_datetime_numpy_dict(datetimes)

    assert len(datetime_features) == 4

    assert len(datetime_features["wind_date_sin"]) == len(datetimes)
    assert (datetime_features["wind_date_cos"] != datetime_features["wind_date_sin"]).all()

    # assert all values are between -1 and 1
    assert all(np.abs(datetime_features["wind_date_sin"]) <= 1)
    assert all(np.abs(datetime_features["wind_date_cos"]) <= 1)
    assert all(np.abs(datetime_features["wind_time_sin"]) <= 1)
    assert all(np.abs(datetime_features["wind_time_cos"]) <= 1)


def test_make_datetime_numpy_batch_custom_key_prefix():
    # Test function correctly applies custom prefix to dict keys
    datetimes = pd.to_datetime(["2024-06-20 12:00", "2024-06-20 12:30", "2024-06-20 13:00"])
    key_prefix = "solar"

    datetime_features = make_datetime_numpy_dict(datetimes, key_prefix=key_prefix)

    # Assert dict contains expected quantity of keys and verify starting with custom prefix
    assert len(datetime_features) == 4
    assert all(key.startswith(key_prefix) for key in datetime_features.keys())


def test_make_datetime_numpy_batch_empty_input():
    # Verification that function raises error for empty input
    datetimes = pd.DatetimeIndex([])

    with pytest.raises(
        ValueError, match="Input datetimes is empty for 'make_datetime_numpy_dict' function"
    ):
        make_datetime_numpy_dict(datetimes)
