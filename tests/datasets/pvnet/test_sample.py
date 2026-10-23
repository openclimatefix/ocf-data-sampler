import numpy as np
import pandas as pd
import pytest
from torch.utils.data import default_collate

from ocf_data_sampler.config.model import (
    DatetimeEncoding,
    PVNetDataConfig,
    SamplingGrid,
)
from ocf_data_sampler.datasets.pvnet.sample import (
    _build_feature_datetimes,
    _build_solar_features,
    add_source_arrays,
    add_source_metadata,
    build_numpy_sample,
)
from ocf_data_sampler.datasets.pvnet.types import Location
from ocf_data_sampler.load.generation import open_generation
from tests.conftest import to_datetime64


@pytest.mark.parametrize("include_extra_metadata", [False, True])
def test_build_numpy_sample_metadata(include_extra_metadata):
    config = PVNetDataConfig(
        sampling_grid=SamplingGrid(
            locations_csv_path="locations.csv",
            t0_resolution_minutes=30
        ),
    )
    sample = build_numpy_sample(
        source_data={},
        t0=np.datetime64("2024-01-01"),
        location=Location(7, -1.0, 52.0),
        config=config,
        include_extra_metadata=include_extra_metadata,
    )
    assert sample["location_id"] == 7
    assert sample["t0"] == 1704067200.0
    assert ("metadata" in sample) == include_extra_metadata

    if include_extra_metadata:
        assert sample["metadata"] == {"location_longitude": -1.0, "location_latitude": 52.0}
        # Check that the default collate works on the metadata
        batch = default_collate([sample, sample])
        np.testing.assert_array_equal(batch["metadata"]["location_longitude"], [-1.0, -1.0])


def test_build_feature_datetimes():
    config = DatetimeEncoding(
        interval_start_minutes=-60, interval_end_minutes=60, time_resolution_minutes=30,
    )
    datetimes = _build_feature_datetimes(np.datetime64("2024-06-20T12:45"), config)

    expected = to_datetime64([
        "2024-06-20T11:30", "2024-06-20T12:00", "2024-06-20T12:30",
        "2024-06-20T13:00", "2024-06-20T13:30",
    ])
    np.testing.assert_array_equal(datetimes, expected)


def test_build_solar_features():
    datetimes = pd.date_range("2024-06-20 12:00", "2024-06-20 16:00", freq="30min").values
    sample = _build_solar_features(datetimes, lon=0, lat=51.5)

    # Assertion accounting for solar coord normalisation
    assert set(sample) == {"solar_elevation", "solar_azimuth"}
    assert sample["solar_elevation"].dtype == np.float32
    assert sample["solar_azimuth"].dtype == np.float32
    assert np.all((sample["solar_elevation"] >= 0) & (sample["solar_elevation"] <= 1))
    assert np.all((sample["solar_azimuth"] >= 0) & (sample["solar_azimuth"] <= 1))


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    """add_source_arrays just extracts generation_mw/capacity_mwp as-is - normalising
    generation_mw to a capacity factor is normalise_source_data's job (see test_preprocess.py).
    """
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)
    numpy_sample = {"location_id": 1, "t0": np.float64(1672531200.0)}
    add_source_arrays(
        numpy_sample, {"generation_input": da, "generation_target": da},
    )

    generation_mw = da.sel(gen_param="generation_mw").values
    capacity = da.sel(gen_param="capacity_mwp").values

    for key in ("generation_input", "generation_target"):
        # Assert structure
        assert isinstance(numpy_sample, dict)
        assert key in numpy_sample
        assert f"{key}_capacity_mwp" in numpy_sample
        assert f"{key}_time_utc" in numpy_sample

        # Assert content is passed through unchanged
        assert np.array_equal(numpy_sample[key], generation_mw)
        assert np.array_equal(numpy_sample[f"{key}_capacity_mwp"], capacity)
        assert isinstance(numpy_sample[f"{key}_time_utc"], np.ndarray)
        assert numpy_sample[f"{key}_time_utc"].dtype == np.float64
        np.testing.assert_array_equal(
            numpy_sample[f"{key}_time_utc"], 1672531200.0 + np.arange(10) * 1800,
        )


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    numpy_sample = {}
    add_source_arrays(numpy_sample, {"nwp": {"ukv": ds_nwp_ukv_time_sliced}})

    assert (numpy_sample["nwp"]["ukv"] == ds_nwp_ukv_time_sliced.values).all()


def test_convert_satellite_to_numpy_sample(da_sat_like):
    numpy_sample = {}
    add_source_arrays(numpy_sample, {"sat": da_sat_like})

    assert (numpy_sample["satellite"] == da_sat_like.values).all()


def test_add_source_metadata(ds_nwp_ukv_time_sliced, da_sat_like):
    da_nwp = ds_nwp_ukv_time_sliced.assign_coords(
        init_time_utc=np.datetime64("2024-01-02"),
    )
    metadata = {}
    add_source_metadata(metadata, {"sat": da_sat_like, "nwp": {"ukv": da_nwp}})

    # Both sources start at 2024-01-02 00:00 UTC.
    start_posix_seconds = 1704153600.0
    assert metadata["satellite_time_utc"][0] == start_posix_seconds
    assert metadata["nwp"]["ukv"]["init_time_utc"] == start_posix_seconds
    np.testing.assert_array_equal(metadata["nwp"]["ukv"]["step_hours"], np.arange(9))
