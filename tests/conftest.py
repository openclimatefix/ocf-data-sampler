from pathlib import Path

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration

# Constants
TEST_DIR = Path(__file__).parent
CONFIG_DIR = TEST_DIR / "fixtures" / "configs"
NWP_FREQ = pd.Timedelta("3h")
RANDOM_SEED = 42

# The LOCATION_IDS catalog mirrors the GSPs: ID 0 is the national aggregate and IDs 1-317 are the
# regional GSPs
LOCATION_IDS = tuple(range(318))
# The SITE_LOCATION_IDS catalog has no national aggregate since they mirror different sites
SITE_LOCATION_IDS = tuple(range(1, 11))

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


def save_csv(df, path, filename):
    """Save dataframe to csv"""
    csv_path = path / filename
    df.to_csv(csv_path, index=False)
    return str(csv_path)


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
    data = dask.array.random.random(
        (len(variables), 288, 100, 100),
        chunks=(-1, 10, -1, -1),
    ).astype(np.float32)
    data[:, 10, :, :] = np.nan

    ds = xr.DataArray(
        data,
        coords={
            "channel": variables,
            "time_utc": pd.date_range("2023-01-01 00:00", "2023-01-01 23:55", freq="5min"),
            "y_geostationary": np.linspace(4191563, 5304712, 100),
            "x_geostationary": np.linspace(15002, -1824245, 100),
        },
        attrs={"area": UK_SAT_AREA},
    ).to_dataset(name="data", promote_attrs=True)

    yield save_zarr(ds, session_tmp_path, "test_sat.zarr")


# NWP datasets
@pytest.fixture(scope="session")
def ds_nwp_ukv(session_rng):
    coords = {
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="180min", periods=24 * 7),
        "variable": ["si10", "dswrf", "t", "prate"],
        "step": pd.timedelta_range("0h", "10h", freq="1h"),
        "x_osgb": np.linspace(-239_000, 857_000, 50),
        "y_osgb": np.linspace(-183_000, 1225_000, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 200, shape).astype(np.float32)
    return create_xr_dataset(coords, data, "UKV")


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(session_tmp_path, ds_nwp_ukv):
    chunks = {
        "init_time_utc": 1,
        "step": -1,
        "variable": -1,
        "x_osgb": 50,
        "y_osgb": 50,
    }
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
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="6h", periods=24 * 7),
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
    chunks = {"init_time_utc": 1, "step": -1, "variable": -1, "longitude": 50, "latitude": 50}
    yield save_zarr(ds_nwp_ecmwf, session_tmp_path, "ecmwf_nwp.zarr", chunks)


@pytest.fixture(scope="session")
def nwp_cloudcasting_zarr_path(session_tmp_path, session_rng):
    coords = {
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="1h", periods=2),
        "variable": ["IR_097", "VIS008", "WV_073"],
        "step": pd.timedelta_range("15min", "180min", freq="15min"),
        "x_geostationary": np.linspace(15002, -1824245, 100),
        "y_geostationary": np.linspace(4191563, 5304712, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = session_rng.uniform(0, 1, shape).astype(np.float32)

    ds = create_xr_dataset(coords, data, "sat_pred", attrs={"area": UK_SAT_AREA})
    chunks = {
        "init_time_utc": 1,
        "step": -1,
        "variable": -1,
        "x_geostationary": 50,
        "y_geostationary": 50,
    }
    yield save_zarr(ds, session_tmp_path, "cloudcasting.zarr", chunks)


def _locations_dataframe(session_rng, location_ids):
    """Build a locations catalog over the given IDs, with random points in a rough UK bbox."""
    lat_min, lat_max = 49.9, 58.7
    lon_min, lon_max = -8.6, 1.8

    return pd.DataFrame(
        {
            "location_id": location_ids,
            "longitude": session_rng.uniform(lon_min, lon_max, len(location_ids)),
            "latitude": session_rng.uniform(lat_min, lat_max, len(location_ids)),
        },
    )


@pytest.fixture(scope="session")
def df_locations(session_rng):
    """The locations catalog - the source of truth for which locations are samplable."""
    return _locations_dataframe(session_rng, LOCATION_IDS)


@pytest.fixture(scope="session")
def df_site_locations(session_rng):
    """The locations catalog for the site-level fixtures."""
    return _locations_dataframe(session_rng, SITE_LOCATION_IDS)


@pytest.fixture(scope="session")
def ds_generation(session_rng, df_locations):
    """Generation for every catalogued location with no missing generation data"""

    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    location_ids = df_locations["location_id"].values
    shape = (len(times), len(location_ids))

    capacity = 200
    capacities = np.full(shape, fill_value=capacity, dtype="float32")
    generations = session_rng.uniform(0, capacity, shape).astype("float32")

    return xr.Dataset(
        data_vars={
            "capacity_mwp": (("time_utc", "location_id"), capacities),
            "generation_mw": (("time_utc", "location_id"), generations),
        },
        coords={
            "time_utc": times,
            "location_id": location_ids,
        },
    )


# location data (non overlapping time periods) and starting with id 1
@pytest.fixture(scope="session")
def ds_site_generation(session_rng, df_site_locations):
    # Define a global time range (covers all possible site periods)
    global_times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    n_times = len(global_times)

    location_ids = df_site_locations["location_id"].values
    n_sites = len(location_ids)

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

    return xr.Dataset(
        data_vars={
            "capacity_mwp": (("time_utc", "location_id"), capacity),
            "generation_mw": (("time_utc", "location_id"), generation),
        },
        coords={
            "time_utc": global_times,
            "location_id": location_ids,
        },
    )


@pytest.fixture(scope="session")
def generation_zarr_path(session_tmp_path, ds_generation):
    yield save_zarr(ds_generation, session_tmp_path, "generation.zarr")


@pytest.fixture(scope="session")
def site_generation_zarr_path(session_tmp_path, ds_site_generation):
    yield save_zarr(ds_site_generation, session_tmp_path, "site_generation.zarr")


@pytest.fixture(scope="session")
def locations_csv_path(session_tmp_path, df_locations):
    yield save_csv(df_locations, session_tmp_path, "locations.csv")


@pytest.fixture(scope="session")
def site_locations_csv_path(session_tmp_path, df_site_locations):
    yield save_csv(df_site_locations, session_tmp_path, "site_locations.csv")


@pytest.fixture()
def pvnet_config_filename(
    tmp_path,
    config_filename,
    nwp_ukv_zarr_path,
    generation_zarr_path,
    locations_csv_path,
    sat_zarr_path,
):
    config = load_yaml_configuration(config_filename)
    config.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.satellite.zarr_path = sat_zarr_path
    config.generation.zarr_path = generation_zarr_path
    config.sampling_grid.locations_csv_path = locations_csv_path

    path = tmp_path / "configuration.yaml"
    save_yaml_configuration(config, str(path))
    return str(path)


@pytest.fixture(scope="session")
def pvnet_site_config_filename(
    session_tmp_path,
    config_filename,
    nwp_ukv_zarr_path,
    site_generation_zarr_path,
    site_locations_csv_path,
    sat_zarr_path,
):
    config = load_yaml_configuration(config_filename)
    config.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.satellite.zarr_path = sat_zarr_path
    config.generation.zarr_path = site_generation_zarr_path
    config.sampling_grid.locations_csv_path = site_locations_csv_path
    # The site catalog has no national aggregate, so nothing to exclude
    config.sampling_grid.exclude_location_ids = []

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
