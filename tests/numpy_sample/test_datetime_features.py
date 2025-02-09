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

    assert len(datetime_features["date_sin"]) == len(datetimes)
    assert (datetime_features["date_cos"] != datetime_features["date_sin"]).all()

    # assert all values are between -1 and 1
    assert all(np.abs(datetime_features["date_sin"]) <= 1)
    assert all(np.abs(datetime_features["date_cos"]) <= 1)
    assert all(np.abs(datetime_features["time_sin"]) <= 1)
    assert all(np.abs(datetime_features["time_cos"]) <= 1)


def test_make_datetime_numpy_batch_empty_input():
    # Verification that function raises error for empty input
    datetimes = pd.DatetimeIndex([])

    with pytest.raises(
        ValueError, match="Input datetimes is empty for 'make_datetime_numpy_dict' function"
    ):
        make_datetime_numpy_dict(datetimes)
