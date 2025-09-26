import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.constants import NWP_FREQ


@pytest.fixture(scope="module")
def da_sat_like():
    """Create dummy data which looks like satellite data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(x), len(y))),
        coords={
            "time_utc": (["time_utc"], datetimes),
            "x_geostationary": (["x_geostationary"], x),
            "y_geostationary": (["y_geostationary"], y),
        },
    )
    return da_sat


@pytest.fixture(scope="module")
def da_nwp_like():
    """Create dummy data which looks like NWP data"""

    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq=NWP_FREQ)
    steps = pd.timedelta_range("0h", "16h", freq="1h")
    channels = ["t", "dswrf"]

    da_nwp = xr.DataArray(
        np.random.normal(size=(len(datetimes), len(steps), len(channels), len(x), len(y))),
        coords={
            "init_time_utc": (["init_time_utc"], datetimes),
            "step": (["step"], steps),
            "channel": (["channel"], channels),
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )
    return da_nwp


@pytest.fixture(scope="module")
def da_sample():

    datetimes = pd.date_range("2024-01-01 12:00", "2024-01-01 13:00", freq="5min")

    da_sat = xr.DataArray(
        np.random.normal(size=(len(datetimes),)),
        coords={"time_utc": (["time_utc"], datetimes)},
    )
    return da_sat


@pytest.fixture(scope="module")
def da():
    # Create dummy data
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)

    da = xr.DataArray(
        np.random.normal(size=(len(x), len(y))),
        coords={
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )
    return da
