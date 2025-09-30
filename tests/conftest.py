import hashlib
from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import Site, SolarPosition
from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset

# Constants
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"
CONFIG_DIR = TEST_DATA_DIR / "configs"

NWP_FREQ = pd.Timedelta("3h")
RANDOM_SEED = 42

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


@pytest.fixture(scope="session")
def session_rng():
    """Session-scoped random number generator for reproducible test data"""
    return np.random.default_rng(RANDOM_SEED)


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


# Config path fixtures
@pytest.fixture()
def test_config_filename():
    return str(CONFIG_DIR / "test_config.yaml")


@pytest.fixture()
def test_config_gsp_path():
    return str(CONFIG_DIR / "gsp_test_config.yaml")


@pytest.fixture(scope="session")
def site_test_config_path():
    return str(CONFIG_DIR / "site_test_config.yaml")


@pytest.fixture(scope="session")
def config_filename():
    return str(CONFIG_DIR / "pvnet_test_config.yaml")


# Helper functions
def create_nwp_dataset(coords_dict, rng, data_range=(0, 200), name="NWP"):
    """Create NWP-style xarray dataset with random data"""
    coords = tuple(coords_dict.items())
    shape = tuple(len(values) for _, values in coords)
    data = rng.uniform(*data_range, size=shape).astype(np.float32)
    return xr.DataArray(data, coords=coords).to_dataset(name=name)


def save_to_zarr(ds, tmp_path, filename, chunk_dict=None):
    """Save dataset to zarr with optional chunking"""
    if chunk_dict:
        ds = ds.chunk(chunk_dict)
    zarr_path = tmp_path / filename
    ds.to_zarr(zarr_path)
    return str(zarr_path)


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):
    variables = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
                 "IR_134", "VIS006", "VIS008", "WV_062", "WV_073"]
    x = np.linspace(start=15002, stop=-1824245, num=100)
    y = np.linspace(start=4191563, stop=5304712, num=100)
    times = pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min")

    data = dask.array.zeros(
        shape=(len(variables), len(times), len(y), len(x)),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
    )
    data[:, 10, :, :] = np.nan

    ds = xr.DataArray(
        data=data,
        coords={"variable": variables, "time": times,
                "y_geostationary": y, "x_geostationary": x},
        attrs={"area": uk_sat_area_string},
    ).to_dataset(name="data")

    yield save_to_zarr(ds, session_tmp_path, "test_sat.zarr")


@pytest.fixture(scope="session")
def ds_nwp_ukv(session_rng):
    coords = {
        "init_time": pd.date_range(start="2023-01-01 00:00", freq="180min", periods=24 * 7),
        "variable": ["si10", "dswrf", "t", "prate"],
        "step": pd.timedelta_range("0h", "10h", freq="1h"),
        "x": np.linspace(-239_000, 857_000, 50),
        "y": np.linspace(-183_000, 1225_000, 100),
    }
    return create_nwp_dataset(coords, session_rng, name="UKV")


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(session_tmp_path, ds_nwp_ukv):
    chunk_dict = {"init_time": 1, "step": -1, "variable": -1, "x": 50, "y": 50}
    yield save_to_zarr(ds_nwp_ukv, session_tmp_path, "ukv_nwp.zarr", chunk_dict)


@pytest.fixture()
def ds_nwp_ukv_time_sliced(session_rng):
    t0 = pd.to_datetime("2024-01-02 00:00")
    steps = pd.timedelta_range("0h", "8h", freq="1h")

    coords = {
        "step": (["step"], steps),
        "channel": (["channel"], ["t", "dswrf"]),
        "x_osgb": (["x_osgb"], np.arange(-100, 100, 10)),
        "y_osgb": (["y_osgb"], np.arange(-100, 100, 10)),
    }

    shape = (len(steps), 2, 20, 20)
    data = session_rng.normal(size=shape)
    da_nwp = xr.DataArray(data, coords=coords)
    return da_nwp.assign_coords(init_time_utc=("step", [t0 for _ in steps]))


@pytest.fixture(scope="session")
def ds_nwp_ecmwf(session_rng):
    coords = {
        "init_time": pd.date_range(start="2023-01-01 00:00", freq="6h", periods=24 * 7),
        "variable": ["t2m", "dswrf", "mcc"],
        "step": pd.timedelta_range("0h", "14h", freq="1h"),
        "longitude": np.arange(-12.0, 3.0),
        "latitude": np.arange(48.0, 60.0),
    }
    return create_nwp_dataset(coords, session_rng, name="ECMWF_UK")


@pytest.fixture(scope="session")
def nwp_ecmwf_zarr_path(session_tmp_path, ds_nwp_ecmwf):
    chunk_dict = {"init_time": 1, "step": -1, "variable": -1, "longitude": 50, "latitude": 50}
    yield save_to_zarr(ds_nwp_ecmwf, session_tmp_path, "ukv_ecmwf.zarr", chunk_dict)


@pytest.fixture(scope="session")
def icon_eu_zarr_path(session_tmp_path, session_rng):
    latitude = np.linspace(29.5, 35.69, 100)
    longitude = np.linspace(-23.5, -17.31, 100)
    step = pd.timedelta_range("0h", "5D", freq="1h")
    channel_names = np.array(["t_1000hPa", "u_10m", "v_10m"], dtype=np.str_)

    attrs = {
        "Conventions": "CF-1.7",
        "GRIB_centre": "edzw",
        "GRIB_centreDescription": "Offenbach",
        "GRIB_edition": 2,
        "institution": "Offenbach",
    }

    paths = []
    for hour in ["00", "06"]:
        time_utc = pd.Timestamp(f"2021-11-01T{hour}:00:00")
        shape = (len(step), len(channel_names), len(latitude), len(longitude))
        data = session_rng.random(shape).astype(np.float32)

        da = xr.DataArray(
            data=data,
            coords={"step": step, "latitude": latitude, "longitude": longitude,
                    "init_time_utc": time_utc, "channel": channel_names},
            dims=("step", "channel", "latitude", "longitude"),
            attrs=attrs,
        )
        da.coords["valid_time"] = da.init_time_utc + da.step

        zarr_path = save_to_zarr(
            da.to_dataset(name="icon_eu_data"),
            session_tmp_path,
            f"20211101_{hour}.zarr",
        )
        paths.append(zarr_path)

    return paths


@pytest.fixture(scope="session")
def nwp_cloudcasting_zarr_path(session_tmp_path, session_rng):
    coords = {
        "init_time": pd.date_range(start="2023-01-01 00:00", freq="1h", periods=2),
        "variable": ["IR_097", "VIS008", "WV_073"],
        "step": pd.timedelta_range("15min", "180min", freq="15min"),
        "x_geostationary": np.linspace(start=15002, stop=-1824245, num=100),
        "y_geostationary": np.linspace(start=4191563, stop=5304712, num=100),
    }

    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 1, size=shape).astype(np.float32)

    nwp_data = xr.DataArray(
        data,
        coords=tuple(coords.items()),
        attrs={"area": uk_sat_area_string},
    ).to_dataset(name="sat_pred")

    chunk_dict = {"init_time": 1, "step": -1, "variable": -1,
                  "x_geostationary": 50, "y_geostationary": 50}
    yield save_to_zarr(nwp_data, session_tmp_path, "cloudcasting.zarr", chunk_dict)


@pytest.fixture(scope="session")
def ds_uk_gsp(session_rng):
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(0, 318)
    coords = (("datetime_gmt", times), ("gsp_id", gsp_ids))

    capacity = np.ones((len(times), len(gsp_ids)))
    generation = session_rng.uniform(0, 200, size=(len(times), len(gsp_ids))).astype(np.float32)

    return xr.Dataset({
        "capacity_mwp": xr.DataArray(capacity, coords=coords),
        "installedcapacity_mwp": xr.DataArray(capacity, coords=coords),
        "generation_mw": xr.DataArray(generation, coords=coords),
    })


@pytest.fixture(scope="session")
def uk_gsp_zarr_path(session_tmp_path, ds_uk_gsp):
    yield save_to_zarr(ds_uk_gsp, session_tmp_path, "uk_gsp.zarr")


def create_site_data(
    tmp_path_base: Path,
    rng: np.random.Generator,
    num_sites: int = 10,
    start_time_str: str = "2023-01-01 00:00",
    end_time_str: str = "2023-01-02 00:00",
    time_freq: str = "30min",
    site_interval_start_minutes: int = -30,
    site_interval_end_minutes: int = 60,
    site_time_resolution_minutes: int = 30,
) -> Site:
    """Create fake site data with reproducible random generation"""
    param_tuple = (num_sites, start_time_str, end_time_str, time_freq,
                   site_interval_start_minutes, site_interval_end_minutes,
                   site_time_resolution_minutes)
    param_key = hashlib.sha256(str(param_tuple).encode()).hexdigest()

    times = pd.date_range(start_time_str, end_time_str, freq=time_freq)
    site_ids = list(range(num_sites))

    # Base arrays for site properties
    base_arrays = {
        "capacity_kwp": np.array([0.1, 1.1, 4, 6, 8, 9, 15, 2, 3, 5, 7, 10, 12, 1, 0.5]),
        "longitude": np.round(np.linspace(-4, -3, 15), 2),
        "latitude": np.round(np.linspace(51, 52, 15), 2),
    }

    # Slice to match num_sites
    site_props = {k: v[:num_sites] for k, v in base_arrays.items()}

    # Generate data
    generation_data = rng.uniform(0, 200, size=(len(times), num_sites)).astype(np.float32)
    capacity_data = np.tile(site_props["capacity_kwp"], (len(times), 1)).astype(np.float32)

    coords = (("time_utc", times), ("site_id", site_ids))
    generation_ds = xr.Dataset({
        "capacity_kwp": xr.DataArray(capacity_data, coords=coords),
        "generation_kw": xr.DataArray(generation_data, coords=coords),
    })

    meta_df = pd.DataFrame({"site_id": site_ids, **site_props})

    # Save files
    data_path = tmp_path_base / f"sites_data_{param_key}.netcdf"
    metadata_path = tmp_path_base / f"sites_metadata_{param_key}.csv"

    generation_ds.to_netcdf(data_path)
    meta_df.to_csv(metadata_path, index=False)

    return Site(
        file_path=str(data_path),
        metadata_file_path=str(metadata_path),
        interval_start_minutes=site_interval_start_minutes,
        interval_end_minutes=site_interval_end_minutes,
        time_resolution_minutes=site_time_resolution_minutes,
    )


@pytest.fixture(scope="session")
def data_sites(session_tmp_path, session_rng):
    return create_site_data(tmp_path_base=session_tmp_path, rng=session_rng)


@pytest.fixture(scope="session")
def default_data_site_model(data_sites):
    return data_sites


def update_config_paths(config, **zarr_paths):
    """Helper to update config paths"""
    for key, path in zarr_paths.items():
        if key == "nwp_ukv":
            config.input_data.nwp["ukv"].zarr_path = path
        elif key == "satellite":
            config.input_data.satellite.zarr_path = path
        elif key == "gsp":
            config.input_data.gsp.zarr_path = path
    return config


@pytest.fixture()
def pvnet_config_filename(tmp_path, config_filename, nwp_ukv_zarr_path,
                          uk_gsp_zarr_path, sat_zarr_path):
    config = load_yaml_configuration(config_filename)
    config = update_config_paths(
        config,
        nwp_ukv=nwp_ukv_zarr_path,
        satellite=sat_zarr_path,
        gsp=uk_gsp_zarr_path,
    )

    config_path = tmp_path / "configuration.yaml"
    save_yaml_configuration(config, str(config_path))
    return str(config_path)


@pytest.fixture()
def site_config_filename(tmp_path, site_test_config_path, nwp_ukv_zarr_path,
                         sat_zarr_path, default_data_site_model):
    config = load_yaml_configuration(site_test_config_path)
    config = update_config_paths(config, nwp_ukv=nwp_ukv_zarr_path, satellite=sat_zarr_path)
    config.input_data.site = default_data_site_model
    config.input_data.gsp = None
    config.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    config_path = tmp_path / "configuration_site_test.yaml"
    save_yaml_configuration(config, str(config_path))
    yield str(config_path)


@pytest.fixture()
def sites_dataset(site_config_filename):
    return SitesDataset(site_config_filename)


@pytest.fixture(scope="session")
def da_sat_like(session_rng):
    """Create dummy data which looks like satellite data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    datetimes = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq="5min")

    data = session_rng.normal(size=(len(datetimes), len(x), len(y)))
    return xr.DataArray(
        data,
        coords={
            "time_utc": (["time_utc"], datetimes),
            "x_geostationary": (["x_geostationary"], x),
            "y_geostationary": (["y_geostationary"], y),
        },
    )
