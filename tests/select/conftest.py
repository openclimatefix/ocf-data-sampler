import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.conftest import NWP_FREQ


@pytest.fixture(scope="module")
def da_nwp_like():
    """Create dummy NWP-like data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq=NWP_FREQ)
    steps = pd.timedelta_range("0h", "16h", freq="1h")
    channels = ["t", "dswrf"]

    return xr.DataArray(
        np.random.normal(size=(len(datetimes), len(steps), len(channels), len(x), len(y))),
        coords={
            "init_time_utc": (["init_time_utc"], datetimes),
            "step": (["step"], steps),
            "channel": (["channel"], channels),
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )


@pytest.fixture(scope="module")
def da_sample():
    """Create dummy time-series sample data"""
    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")
    return xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords={"time_utc": (["time_utc"], datetimes)},
    )


@pytest.fixture(scope="module")
def da():
    """Create dummy 2D spatial data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    return xr.DataArray(
        np.random.normal(size=(len(x), len(y))),
        coords={
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )
