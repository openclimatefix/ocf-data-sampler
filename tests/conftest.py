from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration

# Constants
RANDOM_SEED = 42

# The LOCATION_IDS catalog mirrors the GSPs: ID 0 is the national aggregate and IDs 1-317 are the
# regional GSPs
LOCATION_IDS = tuple(f"loc_{i}" for i in range(318))

UK_SAT_AREA = {
    "msg_seviri_rss_3km": {
        "projection": {
            "proj": "geos",
            "lon_0": 9.5,
            "h": 35785831,
            "x_0": 0,
            "y_0": 0,
            "a": 6378169,
            "rf": 295.488065897014,
            "no_defs": None,
            "type": "crs",
        },
    },
}


# Helpers
def to_datetime64(datetimes: str | list[str], unit="ns"):
    """Convert a datetime string or list of strings to NumPy datetime64[ns] values."""
    if isinstance(datetimes, str):
        return np.datetime64(datetimes, unit)
    return np.asarray(datetimes, dtype=f"datetime64[{unit}]")


def create_xr_dataset(coords, data, name, attrs=None):
    """Create an xarray dataset with dataset-level attributes."""
    da = xr.DataArray(data, coords=tuple(coords.items()))
    return da.to_dataset(name=name).assign_attrs(attrs or {})


def save_zarr(ds, path, filename, chunks=None):
    """Save dataset to zarr"""
    if chunks:
        ds = ds.chunk(chunks)
    zarr_path = path / filename
    ds.to_zarr(zarr_path)
    return str(zarr_path)


# Core fixtures
@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def config_filename():
    return str(Path(__file__).parent / "fixtures" / "configs" / "pvnet_test_config.yaml")


@pytest.fixture()
def base_config(config_filename):
    """A fresh test configuration, with placeholder data paths."""
    return load_yaml_configuration(config_filename)


@pytest.fixture(scope="session")
def sat_zarr_path(session_tmp_path):
    # IR_016 is the test config's channel and IR_039 is requested in some tests; VIS006 is never
    # requested, so channel selection must drop it
    variables = ["IR_016", "VIS006", "IR_039"]
    rng = np.random.default_rng(RANDOM_SEED)
    data = rng.random((len(variables), 288, 100, 100), dtype=np.float32)
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

    chunks = {"channel": -1, "time_utc": 10, "y_geostationary": -1, "x_geostationary": -1}
    return save_zarr(ds, session_tmp_path, "test_sat.zarr", chunks)


@pytest.fixture(scope="session")
def nwp_ukv_zarr_path(session_tmp_path):
    rng = np.random.default_rng(RANDOM_SEED)
    coords = {
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="180min", periods=24 * 7),
        "variable": ["si10", "dswrf", "t", "prate"],
        "step": pd.timedelta_range("0h", "10h", freq="1h"),
        "x_osgb": np.linspace(-239_000, 857_000, 50),
        "y_osgb": np.linspace(-183_000, 1225_000, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = rng.uniform(0, 200, shape).astype(np.float32)
    ds = create_xr_dataset(coords, data, "UKV")
    chunks = {
        "init_time_utc": 1,
        "step": -1,
        "variable": -1,
        "x_osgb": 50,
        "y_osgb": 50,
    }
    return save_zarr(ds, session_tmp_path, "ukv_nwp.zarr", chunks)


@pytest.fixture(scope="session")
def nwp_ecmwf_zarr_path(session_tmp_path):
    rng = np.random.default_rng(RANDOM_SEED)
    coords = {
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="6h", periods=24 * 7),
        "variable": ["t2m", "dswrf", "mcc"],
        "step": pd.timedelta_range("0h", "14h", freq="1h"),
        "longitude": np.arange(-12.0, 3.0),
        "latitude": np.arange(48.0, 60.0),
    }
    shape = tuple(len(v) for v in coords.values())
    data = rng.uniform(0, 200, shape).astype(np.float32)
    ds = create_xr_dataset(coords, data, "ECMWF_UK")
    chunks = {
        "init_time_utc": 1,
        "step": -1,
        "variable": -1,
        "longitude": 50,
        "latitude": 50,
    }
    return save_zarr(ds, session_tmp_path, "ecmwf_nwp.zarr", chunks)


@pytest.fixture(scope="session")
def nwp_cloudcasting_zarr_path(session_tmp_path):
    rng = np.random.default_rng(RANDOM_SEED)
    coords = {
        "init_time_utc": pd.date_range("2023-01-01 00:00", freq="1h", periods=2),
        "variable": ["IR_097", "VIS008", "WV_073"],
        "step": pd.timedelta_range("15min", "180min", freq="15min"),
        "x_geostationary": np.linspace(15002, -1824245, 100),
        "y_geostationary": np.linspace(4191563, 5304712, 100),
    }
    shape = tuple(len(v) for v in coords.values())
    data = rng.uniform(0, 1, shape).astype(np.float32)

    ds = create_xr_dataset(coords, data, "sat_pred", attrs={"area": UK_SAT_AREA})
    chunks = {
        "init_time_utc": 1,
        "step": -1,
        "variable": -1,
        "x_geostationary": 50,
        "y_geostationary": 50,
    }
    yield save_zarr(ds, session_tmp_path, "cloudcasting.zarr", chunks)


@pytest.fixture(scope="session")
def df_locations():
    """The locations catalog - the source of truth for which locations are samplable.

    Locations are random points in a rough UK bounding box.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    lat_min, lat_max = 49.9, 58.7
    lon_min, lon_max = -8.6, 1.8

    return pd.DataFrame(
        {
            "location_id": LOCATION_IDS,
            "longitude": rng.uniform(lon_min, lon_max, len(LOCATION_IDS)),
            "latitude": rng.uniform(lat_min, lat_max, len(LOCATION_IDS)),
        },
    )


@pytest.fixture(scope="session")
def locations_csv_path(session_tmp_path, df_locations):
    csv_path = session_tmp_path / "locations.csv"
    df_locations.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture(scope="session")
def ds_generation(df_locations):
    """Generation for every catalogued location with no missing generation data"""
    rng = np.random.default_rng(RANDOM_SEED)
    times = pd.date_range("2023-01-01 00:00", "2023-01-02 00:00", freq="30min")
    location_ids = df_locations["location_id"].values
    shape = (len(times), len(location_ids))

    capacity = 200
    capacities = np.full(shape, fill_value=capacity, dtype="float32")
    generations = rng.uniform(0, capacity, shape).astype("float32")

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


@pytest.fixture(scope="session")
def generation_zarr_path(session_tmp_path, ds_generation):
    yield save_zarr(ds_generation, session_tmp_path, "generation.zarr")


@pytest.fixture()
def pvnet_config(
    base_config,
    nwp_ukv_zarr_path,
    generation_zarr_path,
    locations_csv_path,
    sat_zarr_path,
):
    """The test configuration, pointing at the generated test data."""
    base_config.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    base_config.satellite.zarr_path = sat_zarr_path
    base_config.generation.zarr_path = generation_zarr_path
    base_config.sampling_grid.locations_csv_path = locations_csv_path
    return base_config
