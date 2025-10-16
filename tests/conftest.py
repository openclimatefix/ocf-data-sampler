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
CONFIG_DIR = TEST_DIR / "test_data" / "configs"
NWP_FREQ = pd.Timedelta("3h")
RANDOM_SEED = 42

UK_SAT_AREA = """msg_seviri_rss_3km:
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


# Core fixtures
@pytest.fixture(scope="session")
def session_rng():
    """Session-scoped RNG for reproducible test data"""
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


# Helpers
def create_xr_dataset(coords, data, name, attrs=None):
    """Create xarray dataset from coords and data"""
    da = xr.DataArray(data, coords=tuple(coords.items()))
    if attrs:
        da.attrs.update(attrs)
    return da.to_dataset(name=name)


def save_zarr(ds, path, filename, chunks=None):
    """Save dataset to zarr"""
    if chunks:
        ds = ds.chunk(chunks)
    zarr_path = path / filename
    ds.to_zarr(zarr_path)
    return str(zarr_path)


# Satellite data
@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):
    variables = ["IR_016", "IR_039", "IR_087", "IR_097", "IR_108", "IR_120",
                 "IR_134", "VIS006", "VIS008", "WV_062", "WV_073"]
    data = dask.array.zeros(
        (len(variables), 288, 100, 100), chunks=(-1, 10, -1, -1), dtype=np.float32,
    )
    data[:, 10, :, :] = np.nan

    ds = xr.DataArray(
        data,
        coords={
            "variable": variables,
            "time": pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min"),
            "y_geostationary": np.linspace(4191563, 5304712, 100),
            "x_geostationary": np.linspace(15002, -1824245, 100),
        },
        attrs={"area": UK_SAT_AREA},
    ).to_dataset(name="data")

    yield save_zarr(ds, session_tmp_path, "test_sat.zarr")


# NWP datasets
@pytest.fixture(scope="session")
def ds_nwp_ukv(session_rng):
    coords = {
        "init_time": pd.date_range("2023-01-01 00:00", freq="180min", periods=24 * 7),
        "variable": ["si10", "dswrf", "t", "prate"],
        "step": pd.timedelta_range("0h", "10h", freq="1h"),
        "x": np.linspace(-239_000, 857_000, 50),
        "y": np.linspace(-183_000, 1225_000, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 200, shape).astype(np.float32)
    return create_xr_dataset(coords, data, "UKV")


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(session_tmp_path, ds_nwp_ukv):
    chunks = {"init_time": 1, "step": -1, "variable": -1, "x": 50, "y": 50}
    yield save_zarr(ds_nwp_ukv, session_tmp_path, "ukv_nwp.zarr", chunks)


@pytest.fixture()
def ds_nwp_ukv_time_sliced(session_rng):
    steps = pd.timedelta_range("0h", "8h", freq="1h")
    coords = {
        "step": (["step"], steps),
        "channel": (["channel"], ["t", "dswrf"]),
        "x_osgb": (["x_osgb"], np.arange(-100, 100, 10)),
        "y_osgb": (["y_osgb"], np.arange(-100, 100, 10)),
    }
    data = session_rng.normal(size=(len(steps), 2, 20, 20))
    da = xr.DataArray(data, coords=coords)
    t0 = pd.to_datetime("2024-01-02 00:00")
    return da.assign_coords(init_time_utc=("step", [t0] * len(steps)))


@pytest.fixture(scope="session")
def ds_nwp_ecmwf(session_rng):
    coords = {
        "init_time": pd.date_range("2023-01-01 00:00", freq="6h", periods=24 * 7),
        "variable": ["t2m", "dswrf", "mcc"],
        "step": pd.timedelta_range("0h", "14h", freq="1h"),
        "longitude": np.arange(-12.0, 3.0),
        "latitude": np.arange(48.0, 60.0),
    }
    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 200, shape).astype(np.float32)
    return create_xr_dataset(coords, data, "ECMWF_UK")


@pytest.fixture(scope="session")
def nwp_ecmwf_zarr_path(session_tmp_path, ds_nwp_ecmwf):
    chunks = {"init_time": 1, "step": -1, "variable": -1, "longitude": 50, "latitude": 50}
    yield save_zarr(ds_nwp_ecmwf, session_tmp_path, "ukv_ecmwf.zarr", chunks)


@pytest.fixture(scope="session")
def icon_eu_zarr_path(session_tmp_path, session_rng):
    step = pd.timedelta_range("0h", "5D", freq="1h")
    channels = np.array(["t_1000hPa", "u_10m", "v_10m"], dtype=str)
    lat = np.linspace(29.5, 35.69, 100)
    lon = np.linspace(-23.5, -17.31, 100)

    attrs = {"Conventions": "CF-1.7", "GRIB_centre": "edzw",
             "GRIB_centreDescription": "Offenbach", "GRIB_edition": 2,
             "institution": "Offenbach"}

    paths = []
    for hour in ["00", "06"]:
        data = session_rng.random((len(step), len(channels), len(lat), len(lon))).astype(np.float32)
        time_utc = pd.Timestamp(f"2021-11-01T{hour}:00:00")

        da = xr.DataArray(
            data,
            coords={"step": step, "channel": channels, "latitude": lat,
                    "longitude": lon, "init_time_utc": time_utc},
            dims=("step", "channel", "latitude", "longitude"),
            attrs=attrs,
        )
        da.coords["valid_time"] = da.init_time_utc + da.step

        paths.append(save_zarr(da.to_dataset(name="icon_eu_data"),
                              session_tmp_path, f"20211101_{hour}.zarr"))

    return paths


@pytest.fixture(scope="session")
def nwp_cloudcasting_zarr_path(session_tmp_path, session_rng):
    coords = {
        "init_time": pd.date_range("2023-01-01 00:00", freq="1h", periods=2),
        "variable": ["IR_097", "VIS008", "WV_073"],
        "step": pd.timedelta_range("15min", "180min", freq="15min"),
        "x_geostationary": np.linspace(15002, -1824245, 100),
        "y_geostationary": np.linspace(4191563, 5304712, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 1, shape).astype(np.float32)

    ds = create_xr_dataset(coords, data, "sat_pred", attrs={"area": UK_SAT_AREA})
    chunks = {"init_time": 1, "step": -1, "variable": -1,
              "x_geostationary": 50, "y_geostationary": 50}
    yield save_zarr(ds, session_tmp_path, "cloudcasting.zarr", chunks)


# GSP data
@pytest.fixture(scope="session")
def ds_uk_gsp(session_rng):
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    gsp_ids = np.arange(318)
    coords = (("datetime_gmt", times), ("gsp_id", gsp_ids))

    capacity = np.ones((len(times), len(gsp_ids)))
    generation = session_rng.uniform(0, 200, (len(times), len(gsp_ids))).astype(np.float32)

    return xr.Dataset({
        "capacity_mwp": xr.DataArray(capacity, coords=coords),
        "installedcapacity_mwp": xr.DataArray(capacity, coords=coords),
        "generation_mw": xr.DataArray(generation, coords=coords),
    })


@pytest.fixture(scope="session")
def uk_gsp_zarr_path(session_tmp_path, ds_uk_gsp):
    yield save_zarr(ds_uk_gsp, session_tmp_path, "uk_gsp.zarr")


# Site data
def create_site_data(
    tmp_path: Path,
    rng: np.random.Generator,
    num_sites: int = 10,
    start_time: str = "2023-01-01 00:00",
    end_time: str = "2023-01-02 00:00",
    freq: str = "30min",
    interval_start: int = -30,
    interval_end: int = 60,
    time_resolution: int = 30,
    variable_capacity: bool = False,
) -> Site:
    """Create fake site data with reproducible random generation"""
    params = (num_sites, start_time, end_time, freq, interval_start, interval_end, time_resolution)
    key = hashlib.sha256(str(params).encode()).hexdigest()

    times = pd.date_range(start_time, end_time, freq=freq)
    site_ids = list(range(num_sites))

    base = {
        "capacity_kwp": np.full(num_sites, 1),
        "longitude": np.round(np.linspace(-4, -3, num_sites), 2),
        "latitude": np.round(np.linspace(51, 52, num_sites), 2),
    }

    coords = (("time_utc", times), ("site_id", site_ids))
    if variable_capacity:
        ds = xr.Dataset({
            "capacity_kwp": xr.DataArray(
                np.tile(rng.uniform(1,100,1)*base["capacity_kwp"], (len(times), 1)).astype(np.float32), coords=coords,
            ),
            "generation_kw": xr.DataArray(
                rng.uniform(0, 200, (len(times), num_sites)).astype(np.float32), coords=coords,
            ),
        })
    else:
        ds = xr.Dataset({
            "generation_kw": xr.DataArray(
                rng.uniform(0, 200, (len(times), num_sites)).astype(np.float32), coords=coords,
            ),
        })

    data_path = tmp_path / f"sites_data_{key}.netcdf"
    meta_path = tmp_path / f"sites_metadata_{key}.csv"

    ds.to_netcdf(data_path)
    pd.DataFrame({"site_id": site_ids, **base}).to_csv(meta_path, index=False)

    return Site(
        file_path=str(data_path),
        metadata_file_path=str(meta_path),
        interval_start_minutes=interval_start,
        interval_end_minutes=interval_end,
        time_resolution_minutes=time_resolution,
    )


@pytest.fixture(scope="session")
def default_data_site_model(session_tmp_path, session_rng):
    return create_site_data(session_tmp_path, session_rng)


@pytest.fixture(scope="session")
def default_data_site_model_variable_capacity(session_tmp_path, session_rng):
    return create_site_data(session_tmp_path, session_rng, variable_capacity=True)



# Config fixtures
def update_config(config, **paths):
    """Update config with zarr paths"""
    mapping = {"nwp_ukv": ("nwp", "ukv"), "satellite": ("satellite",), "gsp": ("gsp",)}
    for key, path in paths.items():
        obj = config.input_data
        for attr in mapping[key][:-1]:
            obj = getattr(obj, attr)
        if key == "nwp_ukv":
            obj["ukv"].zarr_path = path
        else:
            setattr(obj, mapping[key][-1], type(getattr(obj, mapping[key][-1]))(zarr_path=path)
                    if path else None)
    return config


@pytest.fixture()
def pvnet_config_filename(tmp_path, config_filename, nwp_ukv_zarr_path,
                          uk_gsp_zarr_path, sat_zarr_path):
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.gsp.zarr_path = uk_gsp_zarr_path

    path = tmp_path / "configuration.yaml"
    save_yaml_configuration(config, str(path))
    return str(path)


@pytest.fixture()
def site_config_filename(tmp_path, site_test_config_path, nwp_ukv_zarr_path,
                         sat_zarr_path, default_data_site_model):
    config = load_yaml_configuration(site_test_config_path)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.site = default_data_site_model
    config.input_data.gsp = None
    config.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30, interval_start_minutes=-30, interval_end_minutes=60,
    )

    path = tmp_path / "configuration_site_test.yaml"
    save_yaml_configuration(config, str(path))
    yield str(path)


@pytest.fixture()
def sites_dataset(site_config_filename):
    return SitesDataset(site_config_filename)


@pytest.fixture(scope="session")
def da_sat_like(session_rng):
    """Create dummy satellite-like data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    times = pd.date_range("2024-01-02 00:00", "2024-01-03 00:00", freq="5min")

    return xr.DataArray(
        session_rng.normal(size=(len(times), len(x), len(y))),
        coords={
            "time_utc": (["time_utc"], times),
            "x_geostationary": (["x_geostationary"], x),
            "y_geostationary": (["y_geostationary"], y),
        },
    )
