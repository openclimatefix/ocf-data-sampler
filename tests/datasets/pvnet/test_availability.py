import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.datasets.pvnet.availability import (
    _filter_times_to_requested_periods,
    build_concurrent_sample_index,
    build_sample_index,
    validate_requested_periods,
)


@pytest.mark.parametrize(
    ("periods", "expected_indexer"),
    [
        ([(None, None)], slice(None, None)),
        ([(None, "2023-01-01T07:00")], slice(None, 3)),
        ([("2023-01-01T12:30", None)], slice(6, None)),
        ([
            ("2023-01-01T06:00", "2023-01-01T07:00"),
            ("2023-01-01T12:00", "2023-01-01T13:00"),
        ], [1, 2, 5, 6]),
    ],
)
def test_filter_times_to_requested_periods(periods, expected_indexer):
    times = np.array([
        "2023-01-01T05:00", "2023-01-01T06:00", "2023-01-01T06:30", "2023-01-01T07:00",
        "2023-01-01T11:00", "2023-01-01T12:00", "2023-01-01T12:30", "2023-01-01T13:00",
    ], dtype="datetime64[ns]")
    filtered = _filter_times_to_requested_periods(times, periods)
    np.testing.assert_array_equal(filtered, times[expected_indexer])


@pytest.mark.parametrize("periods", [
    [("2023-01-02", "2023-01-04"), ("2023-01-01", "2023-01-03")],
    [(None, "2023-01-03"), ("2023-01-02", None)],
])
def test_validate_requested_periods_rejects_overlap(periods):
    with pytest.raises(ValueError, match="must not overlap"):
        validate_requested_periods(periods)


@pytest.mark.parametrize("periods", [
    [("2023-01-01", "2023-01-03"), ("2023-01-03", "2023-01-04")],
    [(None, "2023-01-02"), ("2023-01-02", None)],
])
def test_validate_requested_periods_accepts_adjacent_periods(periods):
    validate_requested_periods(periods)


@pytest.mark.parametrize("periods", [
    [], [("", "2023-01-02")], [("2023-01-02", "2023-01-01")],
])
def test_validate_requested_periods_invalid_bounds(periods):
    with pytest.raises(ValueError):
        validate_requested_periods(periods)


@pytest.fixture
def pvnet_config():
    config = load_yaml_configuration("tests/fixtures/configs/pvnet_test_config.yaml")
    config.nwp = None
    config.satellite.time_resolution_minutes = 30
    return config


@pytest.fixture
def location_ids():
    return [1, 2]


@pytest.fixture
def datetimes():
    return np.datetime64("2023-01-01T00:00", "ns") + np.arange(9) * np.timedelta64(30, "m")


def make_generation(values, location_ids, datetimes):
    return xr.DataArray(
        values,
        dims=("time_utc", "location_id"),
        coords={"time_utc": datetimes, "location_id": location_ids},
    )


def make_satellite(datetimes):
    return xr.DataArray(np.ones(len(datetimes)), dims="time_utc", coords={"time_utc": datetimes})


def _assert_sample_index(index, expected_times, expected_ids):
    np.testing.assert_array_equal(index.t0, np.asarray(expected_times, dtype="datetime64[ns]"))
    np.testing.assert_array_equal(index.location_id, expected_ids)
    assert index.location_id.dtype == np.int64


def test_build_sample_index_unusable_location(pvnet_config, location_ids, datetimes):
    pvnet_config.satellite = None
    values = np.ones((len(datetimes), len(location_ids)))
    sources = {"generation": make_generation(values, location_ids, datetimes)}
    sample_index = build_sample_index(sources, location_ids, pvnet_config, None)
    # Samples use one hour (2 steps before t0) of history and two hours (4 steps after t0) of future
    expected_t0s = datetimes[2:-4]
    _assert_sample_index(sample_index, np.tile(expected_t0s, 2), [1, 1, 1, 2, 2, 2])

    concurrent_index = build_concurrent_sample_index(sources, pvnet_config, None)
    np.testing.assert_array_equal(concurrent_index.t0, expected_t0s)

    values[:, 1] = np.nan
    sources = {"generation": make_generation(values, location_ids, datetimes)}
    with pytest.raises(ValueError, match="No t0 times found for location 2"):
        build_sample_index(sources, location_ids, pvnet_config, None)
    with pytest.raises(ValueError, match="every requested location"):
        build_concurrent_sample_index(sources, pvnet_config, None)


def test_build_sample_index_non_overlapping_sources(
    pvnet_config, location_ids, datetimes,
):
    values = np.ones((len(datetimes), len(location_ids)))
    sources = {
        "generation": make_generation(values, location_ids, datetimes),
        "sat": make_satellite(datetimes + np.timedelta64(1, "D")),
    }
    with pytest.raises(ValueError, match="No intersecting time periods found for location 1"):
        build_sample_index(sources, location_ids, pvnet_config, None)
    with pytest.raises(ValueError, match="every requested location"):
        build_concurrent_sample_index(sources, pvnet_config, None)


def test_build_sample_index_generation_only(pvnet_config, location_ids, datetimes):
    pvnet_config.satellite = None
    values = np.ones((len(datetimes), len(location_ids)))

    # Samples are configured to use 2 stamps before t0 and 4 timestamps after t0
    # Location 1 has NaN generation at first timestamp so t0s start from datetime[3]
    values[0, 0] = np.nan
    expected_datetimes_1 = datetimes[3:-4]

    # Location 2 has NaN generation at last timestamp so t0s end at datetime[-5]
    values[-1, 1] = np.nan
    expected_datetimes_2 = datetimes[2:-5]

    sources = {"generation": make_generation(values, location_ids, datetimes)}
    index = build_sample_index(sources, location_ids, pvnet_config, None)
    _assert_sample_index(
        index, np.concatenate([expected_datetimes_1, expected_datetimes_2]), [1, 1, 2, 2],
    )

    # Concurrent samples require all locations to have generation data for each t0
    concurrent = build_concurrent_sample_index(sources, pvnet_config, None)
    np.testing.assert_array_equal(concurrent.t0, datetimes[3:-5])


def test_build_sample_index_no_shared_t0s(pvnet_config, location_ids, datetimes):
    pvnet_config.satellite = None
    values = np.ones((len(datetimes), len(location_ids)))

    # Samples are configured to use 2 stamps before t0 and 4 timestamps after t0
    # Note `datetimes` is regularly spaced with length 9
    values[-2:, 0] = np.nan
    expected_datetimes_1 = datetimes[2:3]

    values[:2, 1] = np.nan
    expected_datetimes_2 = datetimes[4:5]

    # There is no intersection between the t0s expected for the two locations
    assert len(np.intersect1d(expected_datetimes_1, expected_datetimes_2))==0

    # No t0 overlap between location is fine for the regular PVNet sample index
    sources = {"generation": make_generation(values, location_ids, datetimes)}
    index = build_sample_index(sources, location_ids, pvnet_config, None)
    _assert_sample_index(
        index, np.concatenate([expected_datetimes_1, expected_datetimes_2]), [1, 2],
    )

    # The concurrent PVNet sample index finds no t0 times where all locations are available so it
    # raises
    with pytest.raises(ValueError, match="every requested location"):
        build_concurrent_sample_index(sources, pvnet_config, None)


def test_build_sample_index_without_generation(pvnet_config, location_ids, datetimes):
    pvnet_config.generation = None
    sources = {"sat": make_satellite(datetimes)}
    ordinary = build_sample_index(sources, location_ids, pvnet_config, None)
    concurrent = build_concurrent_sample_index(sources, pvnet_config, None)
    # Sample is congigured to use satellite between -30 minutes and 0. So every t0 after the first
    # datetime is available
    expected_t0s = datetimes[1:]
    _assert_sample_index(
        ordinary, np.tile(expected_t0s, len(location_ids)),
        np.repeat(location_ids, len(expected_t0s)),
    )
    np.testing.assert_array_equal(concurrent.t0, expected_t0s)


def test_build_sample_index_without_sources(pvnet_config, location_ids, datetimes):
    pvnet_config.generation = None
    pvnet_config.satellite = None

    requested_periods=[
        ("2023-01-01T00:10", "2023-01-01T01:00"),
        ("2023-01-01T01:00", "2023-01-01T01:30"),
    ]
    index = build_sample_index({}, location_ids, pvnet_config, requested_periods)
    _assert_sample_index(index, np.tile(datetimes[1:3], 2), [1, 1, 2, 2])


@pytest.mark.parametrize("periods", [None, [(None, "2023-01-02")]])
def test_build_sample_index_without_sources_requires_bounds(pvnet_config, location_ids, periods):
    # If no data sources are used (i.e. only solar coords, datetimes, and other metadata) the the
    # requested_periods must be provided. Else the length of the dataset is unbounded
    pvnet_config.generation = None
    pvnet_config.satellite = None
    with pytest.raises(ValueError, match="finite start and end"):
        build_sample_index({}, location_ids, pvnet_config, requested_periods=periods)
