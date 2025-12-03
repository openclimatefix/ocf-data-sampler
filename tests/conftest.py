from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration

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
    data = dask.array.zeros(
        (len(variables), 288, 100, 100),
        chunks=(-1, 10, -1, -1),
        dtype=np.float32,
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

    attrs = {
        "Conventions": "CF-1.7",
        "GRIB_centre": "edzw",
        "GRIB_centreDescription": "Offenbach",
        "GRIB_edition": 2,
        "institution": "Offenbach",
    }

    paths = []
    for hour in ["00", "06"]:
        data = session_rng.random((len(step), len(channels), len(lat), len(lon))).astype(np.float32)
        time_utc = pd.Timestamp(f"2021-11-01T{hour}:00:00")

        da = xr.DataArray(
            data,
            coords={
                "step": step,
                "channel": channels,
                "longitude": lon,
                "latitude": lat,
                "init_time_utc": time_utc,
            },
            dims=("step", "channel", "longitude", "latitude"),
            attrs=attrs,
        )
        da.coords["valid_time"] = da.init_time_utc + da.step

        paths.append(
            save_zarr(
                da.to_dataset(name="icon_eu_data"),
                session_tmp_path,
                f"20211101_{hour}.zarr",
            ),
        )

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
    chunks = {
        "init_time": 1,
        "step": -1,
        "variable": -1,
        "x_geostationary": 50,
        "y_geostationary": 50,
    }
    yield save_zarr(ds, session_tmp_path, "cloudcasting.zarr", chunks)


@pytest.fixture(scope="session")
def ds_generation(session_rng):
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    location_ids = np.arange(318)
    # Rough UK bounding box
    lat_min, lat_max = 49.9, 58.7
    lon_min, lon_max = -8.6, 1.8

    # Generate random uniform points
    longitudes = session_rng.uniform(lon_min, lon_max, len(location_ids)).astype("float64")
    latitudes = session_rng.uniform(lat_min, lat_max, len(location_ids)).astype("float64")

    capacity = np.ones((len(times), len(location_ids)))

    generation = session_rng.uniform(0, 200, (len(times), len(location_ids))).astype(np.float32)

    # Build Dataset
    return xr.Dataset(
        data_vars={
            "capacity_mwp": (("time_utc", "location_id"), capacity),
            "generation_mw": (("time_utc", "location_id"), generation),
        },
        coords={
            "time_utc": times,
            "location_id": location_ids,
            "longitude": ("location_id", longitudes),
            "latitude": ("location_id", latitudes),
        },
    )


# location data (non overlapping time periods) and starting with id 1
@pytest.fixture(scope="session")
def ds_site_generation(session_rng):
    # Define a global time range (covers all possible site periods)
    global_times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    n_times = len(global_times)

    location_ids = np.arange(1, 11)
    n_sites = len(location_ids)

    # Rough UK bounding box
    lat_min, lat_max = 49.9, 58.7
    lon_min, lon_max = -8.6, 1.8

    longitudes = session_rng.uniform(lon_min, lon_max, n_sites).astype("float64")
    latitudes = session_rng.uniform(lat_min, lat_max, n_sites).astype("float64")

    # Initialize with NaNs
    capacity = np.full((n_times, n_sites), np.nan, dtype="float32")
    generation = np.full((n_times, n_sites), np.nan, dtype="float32")

    # Each location gets its own time window (at least 5 hours = 10 half-hour intervals)
    min_length = 10
    for i, _ in enumerate(location_ids):
        start_idx = session_rng.integers(0, n_times - min_length)
        max_possible_end = n_times
        end_idx = session_rng.integers(start_idx + min_length, max_possible_end)
        active_slice = slice(start_idx, end_idx)

        # Fill only active period with random data
        capacity[active_slice, i] = 1.0
        generation[active_slice, i] = session_rng.uniform(0, 200, end_idx - start_idx).astype(
            "float32",
        )

    # Build Dataset
    return xr.Dataset(
        data_vars={
            "capacity_mwp": (("time_utc", "location_id"), capacity),
            "generation_mw": (("time_utc", "location_id"), generation),
        },
        coords={
            "time_utc": global_times,
            "location_id": location_ids,
            "longitude": ("location_id", longitudes),
            "latitude": ("location_id", latitudes),
        },
    )


@pytest.fixture(scope="session")
def generation_zarr_path(session_tmp_path, ds_generation):
    yield save_zarr(ds_generation, session_tmp_path, "generation.zarr")


@pytest.fixture(scope="session")
def site_generation_zarr_path(session_tmp_path, ds_site_generation):
    yield save_zarr(ds_site_generation, session_tmp_path, "site_generation.zarr")


@pytest.fixture()
def pvnet_config_filename(
    tmp_path,
    config_filename,
    nwp_ukv_zarr_path,
    generation_zarr_path,
    sat_zarr_path,
):
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.generation.zarr_path = generation_zarr_path

    path = tmp_path / "configuration.yaml"
    save_yaml_configuration(config, str(path))
    return str(path)


@pytest.fixture(scope="session")
def pvnet_site_config_filename(
    session_tmp_path,
    config_filename,
    nwp_ukv_zarr_path,
    site_generation_zarr_path,
    sat_zarr_path,
):
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.generation.zarr_path = site_generation_zarr_path

    path = session_tmp_path / "configuration.yaml"
    save_yaml_configuration(config, str(path))
    return str(path)


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
