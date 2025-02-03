import pytest

import os
import numpy as np
import pandas as pd
import xarray as xr
import dask.array

from ocf_data_sampler.config.model import Site
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration


_top_test_directory = os.path.dirname(os.path.realpath(__file__))

@pytest.fixture()
def test_config_filename():
    return f"{_top_test_directory}/test_data/configs/test_config.yaml"


@pytest.fixture(scope="session")
def config_filename():
    return f"{_top_test_directory}/test_data/configs/pvnet_test_config.yaml"


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):

    # Define coords for satellite-like dataset
    variables = [
        'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 
        'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073',
    ]
    x = np.linspace(start=15002, stop=-1824245, num=100)
    y = np.linspace(start=4191563, stop=5304712, num=100)
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")
    
    area_string = (
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
    
    # Create satellite-like data with some NaNs
    data = dask.array.zeros(
        shape=(len(variables), len(times), len(y), len(x)), 
        chunks=(-1, 10, -1, -1),
        dtype=np.float32
    )
    data [:, 10, :, :] = np.nan
    
    ds = xr.DataArray(
        data=data,
        coords=dict(
            variable=variables,
            time=times,
            y_geostationary=y,
            x_geostationary=x,
        ),
        attrs=dict(area=area_string),
    ).to_dataset(name="data")

    # Save temporarily as a zarr
    zarr_path = session_tmp_path / "test_sat.zarr"
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
def nwp_ukv_zarr_path(session_tmp_path, ds_nwp_ukv):
    ds = ds_nwp_ukv.chunk(
        {
            "init_time": 1, 
            "step": -1, 
            "variable": -1,
            "x": 50, 
            "y": 50,
        }
    )
    zarr_path = session_tmp_path / "ukv_nwp.zarr"
    ds.to_zarr(zarr_path)
    yield zarr_path


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
def nwp_ecmwf_zarr_path(session_tmp_path, ds_nwp_ecmwf):
    ds = ds_nwp_ecmwf.chunk(
        {
            "init_time": 1, 
            "step": -1, 
            "variable": -1,
            "longitude": 50, 
            "latitude": 50,
        }
    )

    zarr_path = session_tmp_path / "ukv_ecmwf.zarr"
    ds.to_zarr(zarr_path)
    yield zarr_path


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
def data_sites(session_tmp_path) -> Site:
    """
    Make fake data for sites
    Returns: filename for netcdf file, and csv metadata
    """
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    site_ids = list(range(0,10))
    capacity_kwp_1d = np.array([0.1,1.1,4,6,8,9,15,2,3,4])
    # these are quite specific for the fake satellite data
    longitude = np.arange(-4, -3, 0.1)
    latitude = np.arange(51, 52, 0.1)

    generation = np.random.uniform(0, 200, size=(len(times), len(site_ids))).astype(np.float32)

    # repeat capacity in new dims len(times) times
    capacity_kwp = (np.tile(capacity_kwp_1d, len(times))).reshape(len(times),10)

    coords = (
        ("time_utc", times),
        ("site_id", site_ids),
    )

    da_cap = xr.DataArray(
        capacity_kwp,
        coords=coords,
    )

    da_gen = xr.DataArray(
        generation,
        coords=coords,
    )

    # metadata
    meta_df = pd.DataFrame(columns=[], data = [])
    meta_df['site_id'] = site_ids
    meta_df['capacity_kwp'] = capacity_kwp_1d
    meta_df['longitude'] = longitude
    meta_df['latitude'] = latitude

    generation = xr.Dataset({
        "capacity_kwp": da_cap,
        "generation_kw": da_gen,
    })

    filename = f"{session_tmp_path}/sites.netcdf"
    filename_csv = f"{session_tmp_path}/sites_metadata.csv"
    generation.to_netcdf(filename)
    meta_df.to_csv(filename_csv)

    site = Site(
        file_path=filename,
        metadata_file_path=filename_csv,
        interval_start_minutes=-30,
        interval_end_minutes=60,
        time_resolution_minutes=30,
    )

    yield site


@pytest.fixture(scope="session")
def uk_gsp_zarr_path(session_tmp_path, ds_uk_gsp):
    zarr_path = session_tmp_path / "uk_gsp.zarr"
    ds_uk_gsp.to_zarr(zarr_path)
    yield zarr_path


@pytest.fixture()
def pvnet_config_filename(
    tmp_path, config_filename, nwp_ukv_zarr_path, uk_gsp_zarr_path, sat_zarr_path
):

    # adjust config to point to the zarr file
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.gsp.zarr_path = uk_gsp_zarr_path

    filename = f"{tmp_path}/configuration.yaml"
    save_yaml_configuration(config, filename)
    return filename