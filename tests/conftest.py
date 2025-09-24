import hashlib
import os
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import Site, SolarPosition
from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset

_top_test_directory = os.path.dirname(os.path.realpath(__file__))


uk_sat_area_string = """msg_seviri_rss_3km:
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


@pytest.fixture()
def test_config_filename():
    return f"{_top_test_directory}/test_data/configs/test_config.yaml"

@pytest.fixture()
def test_config_gsp_path():
    return f"{_top_test_directory}/test_data/configs/gsp_test_config.yaml"

@pytest.fixture(scope="session")
def site_test_config_path():
    return f"{_top_test_directory}/test_data/configs/site_test_config.yaml"

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
        "IR_016",
        "IR_039",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
        "VIS006",
        "VIS008",
        "WV_062",
        "WV_073",
    ]
    x = np.linspace(start=15002, stop=-1824245, num=100)
    y = np.linspace(start=4191563, stop=5304712, num=100)
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")

    # Create satellite-like data with some NaNs
    data = dask.array.zeros(
        shape=(len(variables), len(times), len(y), len(x)),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )
    data[:, 10, :, :] = np.nan

    ds = xr.DataArray(
        data=data,
        coords={
            "variable": variables,
            "time": times,
            "y_geostationary": y,
            "x_geostationary": x,
        },
        attrs={"area": uk_sat_area_string},
    ).to_dataset(name="data")

    # Save temporarily as a zarr
    zarr_path = session_tmp_path / "test_sat.zarr"
    ds.to_zarr(zarr_path)

    yield str(zarr_path)

@pytest.fixture(scope="session")
def sat_icechunk_path(session_tmp_path):
    """Create a small, custom local icechunk store with expected dimensions for testing."""
    import icechunk

    variables = [
        "IR_016",
        "IR_039",
        "IR_087",
        "IR_097",
        "IR_108",
        "IR_120",
        "IR_134",
        "VIS006",
        "VIS008",
        "WV_062",
        "WV_073",
    ]
    x = np.linspace(start=15002, stop=-1824245, num=50)
    y = np.linspace(start=4191563, stop=5304712, num=50)
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 02:00", freq="5min")

    # Fill with fake data
    data = dask.array.zeros(
        shape=(len(variables), len(times), len(y), len(x)),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )

    ds = xr.DataArray(
        data=data,
        coords={
            "variable": variables,
            "time": times,
            "y_geostationary": y,
            "x_geostationary": x,
        },
        attrs={"area": uk_sat_area_string},
    ).to_dataset(name="data")

    # Create icechunk store using the correct function name
    icechunk_path = session_tmp_path / "bucket" / "test_sat.icechunk"
    os.makedirs(icechunk_path.parent, exist_ok=True)

    storage = icechunk.local_filesystem_storage(str(icechunk_path))
    repo = icechunk.Repository.create(storage)
    session = repo.writable_session("main")

    # Write data to icechunk
    ds.to_zarr(session.store, mode="w")
    session.commit("Initial test data commit")

    yield str(icechunk_path)

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
        },
    )
    zarr_path = session_tmp_path / "ukv_nwp.zarr"
    ds.to_zarr(zarr_path)
    yield str(zarr_path)


@pytest.fixture()
def ds_nwp_ukv_time_sliced():

    t0 = pd.to_datetime("2024-01-02 00:00")
    x = np.arange(-100, 100, 10)
    y = np.arange(-100, 100, 10)
    steps = pd.timedelta_range("0h", "8h", freq="1h")
    channels = ["t", "dswrf"]

    # Create dummy time-sliced NWP data
    da_nwp = xr.DataArray(
        np.random.normal(size=(len(steps), len(channels), len(x), len(y))),
        coords={
            "step": (["step"], steps),
            "channel": (["channel"], channels),
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )

    da_nwp = da_nwp.assign_coords(init_time_utc=("step", [t0 for _ in steps]))

    return da_nwp


@pytest.fixture(scope="session")
def ds_nwp_ecmwf():
    init_times = pd.date_range(start="2023-01-01 00:00", freq="6h", periods=24 * 7)
    steps = pd.timedelta_range("0h", "14h", freq="1h")

    lons = np.arange(-12.0, 3.0)
    lats = np.arange(48.0, 60.0)

    variables = ["t2m", "dswrf", "mcc"]

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("longitude", lons),
        ("latitude", lats),
    )

    nwp_array_shape = tuple(len(coord_values) for _, coord_values in coords)
    nwp_data_raw = np.random.uniform(0, 200, size=nwp_array_shape)

    nwp_data = xr.DataArray(
        nwp_data_raw,
        coords=coords,
    )

    return nwp_data.astype(np.float32).to_dataset(name="ECMWF_UK")


@pytest.fixture(scope="session")
def nwp_ecmwf_zarr_path(session_tmp_path, ds_nwp_ecmwf):
    ds = ds_nwp_ecmwf.chunk(
        {
            "init_time": 1,
            "step": -1,
            "variable": -1,
            "longitude": 50,
            "latitude": 50,
        },
    )

    zarr_path = session_tmp_path / "ukv_ecmwf.zarr"
    ds.to_zarr(zarr_path)
    yield str(zarr_path)


@pytest.fixture(scope="session")
def icon_eu_zarr_path(session_tmp_path):
    date = "20211101"
    hours = ["00", "06"]
    paths = []

    latitude = np.linspace(29.5, 35.69, 100)
    longitude = np.linspace(-23.5, -17.31, 100)
    step = pd.timedelta_range("0h", "5D", freq="1h")

    channel_names = np.array(["t_1000hPa", "u_10m", "v_10m"], dtype=np.str_)

    for hour in hours:
        time_str = f"{date}_{hour}"
        time_utc = pd.Timestamp(f"2021-11-01T{hour}:00:00")

        data_shape = (len(step), len(channel_names), len(latitude), len(longitude))
        data = np.random.rand(*data_shape).astype(np.float32)

        da = xr.DataArray(
            data=data,
            coords={
                "step": step,
                "latitude": latitude,
                "longitude": longitude,
                "init_time_utc": time_utc,
                "channel": channel_names,
            },
            dims=("step", "channel", "latitude", "longitude"),
            attrs={
                "Conventions": "CF-1.7",
                "GRIB_centre": "edzw",
                "GRIB_centreDescription": "Offenbach",
                "GRIB_edition": 2,
                "institution": "Offenbach",
            },
        )

        da.coords["valid_time"] = da.init_time_utc + da.step

        ds_to_save = da.to_dataset(name="icon_eu_data")

        zarr_path = session_tmp_path / f"{time_str}.zarr"
        ds_to_save.to_zarr(zarr_path)
        paths.append(str(zarr_path))

    return paths


@pytest.fixture(scope="session")
def nwp_cloudcasting_zarr_path(session_tmp_path):

    init_times = pd.date_range(start="2023-01-01 00:00", freq="1h", periods=2)
    steps = pd.timedelta_range("15min", "180min", freq="15min")

    variables = ["IR_097", "VIS008", "WV_073"]
    x = np.linspace(start=15002, stop=-1824245, num=100)
    y = np.linspace(start=4191563, stop=5304712, num=100)

    coords = (
        ("init_time", init_times),
        ("variable", variables),
        ("step", steps),
        ("x_geostationary", x),
        ("y_geostationary", y),
    )

    nwp_array_shape = tuple(len(coord_values) for _, coord_values in coords)

    nwp_data = xr.DataArray(
        np.random.uniform(0, 1, size=nwp_array_shape).astype(np.float32),
        coords=coords,
        attrs={"area": uk_sat_area_string},
    ).to_dataset(name="sat_pred")

    zarr_path = session_tmp_path / "cloudcasting.zarr"
    nwp_data.chunk(
        {
            "init_time": 1,
            "step": -1,
            "variable": -1,
            "x_geostationary": 50,
            "y_geostationary": 50,
        },
    ).to_zarr(zarr_path)
    yield str(zarr_path)


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

    return xr.Dataset(
        {
            "capacity_mwp": da_cap,
            "installedcapacity_mwp": da_cap,
            "generation_mw": da_gen,
        },
    )


def create_site_data(
    tmp_path_base: Path,
    num_sites: int = 10,
    start_time_str: str = "2023-01-01 00:00",
    end_time_str: str = "2023-01-02 00:00",
    time_freq: str = "30min",
    site_interval_start_minutes: int = -30,
    site_interval_end_minutes: int = 60,
    site_time_resolution_minutes: int = 30,
) -> Site:
    """
    Make fake data for sites
    Returns: filename for netcdf file, and csv metadata
    """
    param_tuple = (num_sites, start_time_str, end_time_str, time_freq,
                   site_interval_start_minutes, site_interval_end_minutes,
                   site_time_resolution_minutes)
    param_key = hashlib.sha256(str(param_tuple).encode()).hexdigest()

    times = pd.date_range(start_time_str, end_time_str, freq=time_freq)
    site_ids = list(range(num_sites))

    base_capacity_kwp_1d = np.array([0.1, 1.1, 4, 6, 8, 9, 15, 2, 3, 5, 7, 10, 12, 1, 0.5])
    base_longitude = np.round(np.linspace(-4, -3, 15), 2)
    base_latitude = np.round(np.linspace(51, 52, 15), 2)

    capacity_kwp_1d = base_capacity_kwp_1d[:num_sites]
    longitude = base_longitude[:num_sites]
    latitude = base_latitude[:num_sites]

    data_shape = (len(times), num_sites)
    generation_data = np.random.uniform(0, 200, size=data_shape).astype(np.float32)
    capacity_kwp_data = np.tile(capacity_kwp_1d, (len(times), 1)).astype(np.float32)

    coords = (("time_utc", times), ("site_id", site_ids))
    da_cap = xr.DataArray(capacity_kwp_data, coords=coords)
    da_gen = xr.DataArray(generation_data, coords=coords)

    meta_df = pd.DataFrame()
    meta_df["site_id"] = site_ids
    meta_df["capacity_kwp"] = capacity_kwp_1d
    meta_df["longitude"] = longitude
    meta_df["latitude"] = latitude

    generation_ds = xr.Dataset(
        {
            "capacity_kwp": da_cap,
            "generation_kw": da_gen,
        },
    )
    filename_data_path = tmp_path_base / f"sites_data_{param_key}.netcdf"
    filename_csv_path = tmp_path_base / f"sites_metadata_{param_key}.csv"
    generation_ds.to_netcdf(filename_data_path)
    meta_df.to_csv(filename_csv_path, index=False)

    site_model = Site(
        file_path=str(filename_data_path),
        metadata_file_path=str(filename_csv_path),
        interval_start_minutes=site_interval_start_minutes,
        interval_end_minutes=site_interval_end_minutes,
        time_resolution_minutes=site_time_resolution_minutes,
    )
    return site_model


@pytest.fixture(scope="session")
def data_sites(session_tmp_path):
    return create_site_data(tmp_path_base=session_tmp_path)


@pytest.fixture(scope="session")
def uk_gsp_zarr_path(session_tmp_path, ds_uk_gsp):
    zarr_path = session_tmp_path / "uk_gsp.zarr"
    ds_uk_gsp.to_zarr(zarr_path)
    yield str(zarr_path)


@pytest.fixture()
def pvnet_config_filename(
    tmp_path,
    config_filename,
    nwp_ukv_zarr_path,
    uk_gsp_zarr_path,
    sat_zarr_path,
):
    # adjust config to point to the zarr file
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.gsp.zarr_path = uk_gsp_zarr_path

    filename = f"{tmp_path}/configuration.yaml"
    save_yaml_configuration(config, filename)
    return filename


@pytest.fixture(scope="session")
def default_data_site_model(data_sites):
    return data_sites


@pytest.fixture()
def site_config_filename(
    tmp_path,
    site_test_config_path,
    nwp_ukv_zarr_path,
    sat_zarr_path,
    default_data_site_model,
):
    # adjust config to point to the zarr file
    config = load_yaml_configuration(site_test_config_path)
    config.input_data.nwp["ukv"].zarr_path = str(nwp_ukv_zarr_path)
    config.input_data.satellite.zarr_path = str(sat_zarr_path)
    config.input_data.site = default_data_site_model
    config.input_data.gsp = None

    config.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    config_output_path = tmp_path / "configuration_site_test.yaml"
    save_yaml_configuration(config, str(config_output_path))
    yield str(config_output_path)


@pytest.fixture()
def sites_dataset(site_config_filename):
    return SitesDataset(site_config_filename)
