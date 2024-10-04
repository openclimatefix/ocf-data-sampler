import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import tempfile

_top_test_directory = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture()
def test_config_filename():
    return f"{_top_test_directory}/test_data/configs/test_config.yaml"


@pytest.fixture(scope="session")
def config_filename():
    return f"{os.path.dirname(os.path.abspath(__file__))}/test_data/configs/pvnet_test_config.yaml"


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

    # add 100,000 to x_geostationary, this to make sure the fix index is within the satellite image
    ds["x_geostationary"] = ds["x_geostationary"] - 200_000

    # Add some NaNs
    ds["data"].values[:, :, 0, 0] = np.nan

    # make sure channel values are strings
    ds["variable"] = ds["variable"].astype(str)

    # add data attrs area
    ds["data"].attrs["area"] = (
        """msg_seviri_rss_3km:
        description: MSG SEVIRI Rapid Scanning Service area definition with 3 km resolution
        projection:
            proj: geos
            lon_0: 9.5
            h: 35785831
            x_0: 0
            y_0: 0
            a: 6378169
            rf: 295.488065897014
            no_defs: null
            type: crs
        shape:
            height: 298
            width: 615
        area_extent:
            lower_left_xy: [28503.830075263977, 5090183.970808983]
            upper_right_xy: [-1816744.1169023514, 4196063.827395439]
            units: m
        """
    )

    # Specifiy chunking
    ds = ds.chunk({"time": 10, "variable": -1, "y_geostationary": -1, "x_geostationary": -1})

    # Save temporarily as a zarr
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_sat.zarr"
        ds.to_zarr(zarr_path)

        yield zarr_path


@pytest.fixture(scope="session")
def ds_nwp_ukv():
    init_times = pd.date_range(start="2023-01-01 00:00", freq="180min", periods=24 * 7)
    steps = pd.timedelta_range("0h", "10h", freq="1h")

    x = np.linspace(-239_000, 857_000, 50)
    y = np.linspace(-183_000, 1225_000, 100)
    variables = ["si10", "dswrf", "t", "prate"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x", x),
        ("y", y),
    )

    nwp_array_shape = tuple(len(coord_values) for _, coord_values in coords)

    nwp_data = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape).astype(np.float32),
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
            "x": 50, 
            "y": 50,
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/ukv_nwp.zarr"
        ds.to_zarr(filename)
        yield filename


@pytest.fixture(scope="session")
def ds_nwp_ecmwf():
    init_times = pd.date_range(start="2023-01-01 00:00", freq="6h", periods=24 * 7)
    steps = pd.timedelta_range("0h", "14h", freq="1h")

    lons = np.arange(-12, 3)
    lats = np.arange(48, 60)
    variables = ["t2m","dswrf", "mcc"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("longitude", lons),
        ("latitude", lats),
    )

    nwp_array_shape = tuple(len(coord_values) for _, coord_values in coords)

    nwp_data = xr.DataArray(
        np.random.uniform(0, 200, size=nwp_array_shape).astype(np.float32),
        coords=coords,
    )
    return nwp_data.to_dataset(name="ECMWF_UK")


@pytest.fixture(scope="session")
def nwp_ecmwf_zarr_path(ds_nwp_ecmwf):
    ds = ds_nwp_ecmwf.chunk(
        {
            "init_time": 1, 
            "step": -1, 
            "variable": -1,
            "longitude": 50, 
            "latitude": 50,
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = tmpdir + "/ukv_ecmwf.zarr"
        ds.to_zarr(filename)
        yield filename


@pytest.fixture(scope="session")
def ds_uk_gsp():
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    capacity = np.ones((len(times), len(gsp_ids)))
    generation = np.random.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

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

