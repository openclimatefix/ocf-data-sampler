import numpy as np
import pandas as pd
import xarray as xr

import pytest

from ocf_data_sampler.numpy_batch import convert_nwp_to_numpy_batch

from ocf_datapipes.batch import NWPBatchKey

@pytest.fixture(scope="module")
def da_nwp_like():
    """Create dummy data which looks like time-sliced NWP data"""

    t0 = pd.to_datetime("2024-01-02 00:00")

    x = np.arange(-100, 100, 10)
    y = np.arange(-100, 100, 10)
    steps = pd.timedelta_range("0H", "8H", freq="1H")
    target_times = t0 + steps

    channels = ["t", "dswrf"]
    init_times = pd.to_datetime([t0]*len(steps))
    
    # Create dummy time-sliced NWP data
    da_nwp = xr.DataArray(
        np.random.normal(size=(len(target_times), len(channels), len(x), len(y))),
        coords=dict(
            target_times_utc=(["target_times_utc"], target_times),
            channel=(["channel"], channels),
            x_osgb=(["x_osgb"], x),
            y_osgb=(["y_osgb"], y),
        )
    )

    # Add extra non-coordinate dimensions
    da_nwp = da_nwp.assign_coords(
        init_time_utc=("target_times_utc", init_times),
        step=("target_times_utc", steps),
    )

    return da_nwp


def test_convert_nwp_to_numpy_batch(da_nwp_like):

    # Call the function
    numpy_batch = convert_nwp_to_numpy_batch(da_nwp_like)

    # Assert the output type
    assert isinstance(numpy_batch, dict)

    # Assert the shape of the numpy batch
    assert (numpy_batch[NWPBatchKey.nwp] == da_nwp_like.values).all()