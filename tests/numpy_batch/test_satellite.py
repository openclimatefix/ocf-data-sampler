
import numpy as np
import pandas as pd
import xarray as xr

import pytest

from ocf_data_sampler.numpy_batch import convert_satellite_to_numpy_batch

from ocf_datapipes.batch import BatchKey


@pytest.fixture(scope="module")
def da_sat_like():
    """Create dummy data which looks like satellite data"""
    x = np.arange(-100, 100, 10)
    y = np.arange(-100, 100, 10)
    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 12:30", freq="5min")
    channels = ["VIS008", "IR016"]

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(channels), len(x), len(y))),
        coords=dict(
            time_utc=(["time_utc"], datetimes),
            channel=(["channel"], channels),
            x_geostationary=(["x_geostationary"], x),
            y_geostationary=(["y_geostationary"], y),
        )
    )
    return da_sat


def test_convert_satellite_to_numpy_batch(da_sat_like):

    # Call the function
    numpy_batch = convert_satellite_to_numpy_batch(da_sat_like)

    # Assert the output type
    assert isinstance(numpy_batch, dict)

    # Assert the shape of the numpy batch
    assert (numpy_batch[BatchKey.satellite_actual] == da_sat_like.values).all()