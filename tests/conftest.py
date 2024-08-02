import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import tempfile



@pytest.fixture(scope="session")
def config_filename():
    return f"{os.path.dirname(os.path.abspath(__file__))}/test_data/pvnet_test_config.yaml"

@pytest.fixture(scope="session")
def sat_zarr_path():

    # Load dataset which only contains coordinates, but no data
    ds = xr.open_zarr(
        f"{os.path.dirname(os.path.abspath(__file__))}/test_data/non_hrv_shell.zarr.zip"
    ).compute()

    # Add time coord
    ds = ds.assign_coords(time=pd.date_range("2023-01-01 00:00", "2023-01-02 23:55", freq="5min"))

    # Add data to dataset
    ds["data"] = xr.DataArray(
        np.zeros([len(ds[c]) for c in ds.coords], dtype=np.float32),
        coords=ds.coords,
    )

    # Transpose to variables, time, y, x (just in case)
    ds = ds.transpose("variable", "time", "y_geostationary", "x_geostationary")

    # Add some NaNs
    ds["data"].values[:, :, 0, 0] = np.nan

    # Specifiy chunking
    ds = ds.chunk({"time": 10, "variable": -1, "y_geostationary": -1, "x_geostationary": -1})

    # Save temporarily as a zarr
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_sat.zarr"
        ds.to_zarr(zarr_path)

        yield zarr_path


@pytest.fixture(scope="session")
def ds_nwp_ukv():
    init_times = pd.date_range(start="2022-09-01 00:00", freq="180min", periods=24 * 7)
    steps = pd.timedelta_range("0h", "10h", freq="1h")


    # This is much faster:
    x = np.linspace(-239_000, 857_000, 100)
    y = np.linspace(-183_000, 1225_000, 100)[::-1]  # UKV data must run top to bottom
    variables = ["si10", "dswrf", "t", "prate"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    nwp_array_shape = (len(init_times), len(variables), len(steps), len(x), len(y))

    nwp_data = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape),
        coords=coords,
    )
    return nwp_data.to_dataset(name="UKV")


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(ds_nwp_ukv):
    ds = ds_nwp_ukv.chunk(
        {
            "init_time": 1, 
            "step": -1, 
            "variable": -1,
            "x": 100, 
            "y": 100}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/ukv_nwp.zarr"
        ds.to_zarr(filename)
        yield filename


@pytest.fixture(scope="session")
def ds_uk_gsp():
    times = pd.date_range("2022-09-01 00:00", "2022-09-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids)))

    coords = (
        ("datetime_gmt", times),
        ("gsp_id", gsp_ids),
    )


    da_cap = xr.DataArray(
        capacity,
        coords=coords,
    )

    da_gen = xr.DataArray(
        generation,
        coords=coords,
    )

    return xr.Dataset({
        "capacity_mwp": da_cap, 
        "installedcapacity_mwp": da_cap, 
        "generation_mw":da_gen
    })

@pytest.fixture(scope="session")
def uk_gsp_zarr_path(ds_uk_gsp):

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/uk_gsp.zarr"
        ds_uk_gsp.to_zarr(filename)
        yield filename

