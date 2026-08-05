import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.datasets.pvnet.dataset import (
    PVNetConcurrentDataset,
    PVNetDataset,
    get_locations,
    get_time_periods_mask,
)
from tests.conftest import LOCATION_IDS, SITE_LOCATION_IDS


def _pvnet_dataset_sample_check(sample, config, batch_dim = None):
    """Helper function to verify samples"""

    if batch_dim is None:
        batch_dim = ()

    assert isinstance(sample, dict)

    # Specific keys should always be present
    required_keys = [
        "nwp_ukv", "satellite", "generation_input", "generation_target", "t0", "t0_embedding",
    ]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if config.solar_position is not None:
        # Test solar position keys are present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from config
        expected_time_steps = (
            config.solar_position.interval_end_minutes
            - config.solar_position.interval_start_minutes
        ) // config.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on config
        assert sample["solar_azimuth"].shape == (*batch_dim, expected_time_steps)
        assert sample["solar_elevation"].shape == (*batch_dim, expected_time_steps)
    else:
        # Assert that solar position keys are not present
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    datetime_encoding_keys = ["date_sin", "date_cos", "time_sin", "time_cos"]
    if config.datetime_encoding is not None:
        # Test datetime encoding keys are present when configured
        for key in datetime_encoding_keys:
            assert key in sample, f"Datetime encoding key {key} should be present in sample"

        # Get expected time steps from config
        expected_time_steps = (
            config.datetime_encoding.interval_end_minutes
            - config.datetime_encoding.interval_start_minutes
        ) // config.datetime_encoding.time_resolution_minutes + 1

        # Test datetime encoding shapes based on config
        for key in datetime_encoding_keys:
            assert sample[key].shape == (*batch_dim, expected_time_steps)
    else:
        # Assert that datetime encoding keys are not present
        for key in datetime_encoding_keys:
            assert key not in sample, f"Datetime encoding key {key} should not be present"

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite"].shape == (*batch_dim, 7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp_ukv"].shape == (*batch_dim, 4, 1, 2, 2)
    # generation_input: 1 hour of 30 minute data (inclusive) = 3 steps
    # generation_target: 2 hours of 30 minute data (inclusive) = 5 steps
    assert sample["generation_input"].shape == (*batch_dim, 3)
    assert sample["generation_target"].shape == (*batch_dim, 5)
    # The config uses 3 periods each of which generates a sin and cos embedding
    assert sample["t0_embedding"].shape == (*batch_dim, 6)



def test_get_time_periods_mask():
    times = pd.to_datetime([
        "2023-01-01 05:00",
        "2023-01-01 06:00",
        "2023-01-01 06:30",
        "2023-01-01 07:00",
        "2023-01-01 11:00",
        "2023-01-01 12:00",
        "2023-01-01 12:30",
        "2023-01-01 13:00",
    ])

    mask = get_time_periods_mask(
        times,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )
    expected_mask = np.array([False, True, True, True, False, True, True, True])
    assert np.array_equal(mask, expected_mask), f"Expected {expected_mask} but got {mask}"

    mask = get_time_periods_mask(
        times,
        time_periods=[(None, "2023-01-01 07:00")],
    )
    expected_mask = np.array([True, True, True, True, False, False, False, False])
    assert np.array_equal(mask, expected_mask), f"Expected {expected_mask} but got {mask}"


def _expected_num_locations(dataset, catalog_ids):
    """The catalogued locations which survive the config's exclusion list."""
    return len(catalog_ids) - len(dataset.config.sampling_grid.exclude_location_ids)


def test_pvnet_dataset(pvnet_config_filename):
    dataset = PVNetDataset(
        pvnet_config_filename,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )

    expected_t0s = 6  # 2 time periods each with 3 t0s (inclusive) at 30 minute intervals
    num_locs = _expected_num_locations(dataset, LOCATION_IDS)
    assert len(dataset.locations) == num_locs

    assert len(dataset.valid_t0_times) == expected_t0s
    assert len(dataset) == num_locs * expected_t0s

    sample = dataset[0]

    _pvnet_dataset_sample_check(sample, dataset.config)


def test_get_locations_exclude_ids(locations_csv_path):
    excluded_ids = [LOCATION_IDS[0], LOCATION_IDS[5], LOCATION_IDS[-1]]
    locations = get_locations(locations_csv_path, exclude_ids=excluded_ids)

    assert len(locations) == len(LOCATION_IDS) - len(excluded_ids)
    assert not set(excluded_ids) & {loc.id for loc in locations}


def test_get_locations_exclude_unknown_id(locations_csv_path):
    unknown_id = max(LOCATION_IDS) + 1
    with pytest.raises(ValueError, match="not in the locations data"):
        get_locations(locations_csv_path, exclude_ids=[unknown_id])


def test_get_locations_exclude_all_ids(locations_csv_path):
    with pytest.raises(ValueError, match=r"All location IDs .* have been excluded"):
        get_locations(locations_csv_path, exclude_ids=list(LOCATION_IDS))


def test_pvnet_dataset_sites(pvnet_site_config_filename):
    dataset = PVNetDataset(
        pvnet_site_config_filename,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )

    expected_t0s = 6  # 2 time periods each with 3 t0s (inclusive) at 30 minute intervals
    num_locs = _expected_num_locations(dataset, SITE_LOCATION_IDS)
    assert len(dataset.locations) == num_locs
    # Should be less than num_locs * expected_t0s as not all locations have data for all t0s
    # in the time periods
    assert len(dataset.valid_t0_and_location_ids) < num_locs * expected_t0s

    sample = dataset[0]
    _pvnet_dataset_sample_check(sample, dataset.config)


def test_pvnet_dataset_noxarray_mode(pvnet_config_filename):
    dataset = PVNetDataset(pvnet_config_filename, use_xarray=True)
    sample = dataset[0]

    dataset_nox = PVNetDataset(pvnet_config_filename, use_xarray=False)
    sample_nox = dataset_nox[0]

    def check_samples_equal(sample0, sample1):
        assert set(sample0.keys())==set(sample1.keys())
        for k in sample0:
            if isinstance(sample0[k], np.ndarray):
                assert (sample0[k] == sample1[k]).all()
            else:
                assert sample0[k] == sample1[k]

    check_samples_equal(sample, sample_nox)


def test_pvnet_concurrent_dataset(pvnet_config_filename):
    # Create dataset object using limited set of GSPs
    dataset = PVNetConcurrentDataset(pvnet_config_filename)
    num_locations = _expected_num_locations(dataset, LOCATION_IDS)
    assert len(dataset.locations) == num_locations
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 39

    sample = dataset[0]
    _pvnet_dataset_sample_check(sample, dataset.config, (num_locations,))


def test_pvnet_dataset_getitem_bounds(pvnet_config_filename):
    dataset = PVNetDataset(pvnet_config_filename)

    sample_from_last = dataset[len(dataset) - 1]
    sample_from_negative = dataset[-1]
    assert sample_from_negative["t0"] == sample_from_last["t0"]
    assert sample_from_negative["location_id"] == sample_from_last["location_id"]

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]

    with pytest.raises(IndexError):
        _ = dataset[-len(dataset) - 1]


def test_solar_position_decoupling(tmp_path, pvnet_config_filename):
    """Test that solar position calculations are properly decoupled from data sources."""
    config = load_yaml_configuration(pvnet_config_filename)
    config_without_solar = config.model_copy(deep=True)
    config_without_solar.solar_position = None

    # Create version with explicit solar position configuration
    config_with_solar = config.model_copy(deep=True)
    config_with_solar.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=0,
        interval_end_minutes=180,
    )

    # Save both testing configs
    config_without_solar_path = tmp_path / "config_without_solar.yaml"
    config_with_solar_path = tmp_path / "config_with_solar.yaml"
    save_yaml_configuration(config_without_solar, config_without_solar_path)
    save_yaml_configuration(config_with_solar, config_with_solar_path)

    # Create datasets with both configs
    dataset_without_solar = PVNetDataset(config_without_solar_path)
    dataset_with_solar = PVNetDataset(config_with_solar_path)

    # Generate samples
    sample_without_solar = dataset_without_solar[0]
    sample_with_solar = dataset_with_solar[0]

    # Assert solar position keys are only in sample specifically with solar config
    solar_keys = ["solar_azimuth", "solar_elevation"]

    for key in solar_keys:
        assert key not in sample_without_solar, f"Solar key {key} should not be in sample"
    for key in solar_keys:
        assert key in sample_with_solar, f"Solar key {key} should be in sample"


def test_pvnet_dataset_without_generation(tmp_path, pvnet_config_filename):
    """Test that a dataset can be built with locations/NWP/satellite but no generation at all."""
    config = load_yaml_configuration(pvnet_config_filename)
    config.generation = None

    config_path = tmp_path / "config_without_generation.yaml"
    save_yaml_configuration(config, config_path)

    dataset = PVNetDataset(config_path)

    # With no generation data, there's nothing to be incomplete about
    assert dataset.complete_generation

    # All locations from the locations catalog are available - none to filter out
    assert len(dataset.locations) == _expected_num_locations(dataset, LOCATION_IDS)

    sample = dataset[0]
    assert "generation_input" not in sample
    assert "generation_target" not in sample
    assert "nwp_ukv" in sample
    assert "satellite" in sample


def test_pvnet_dataset_raw_sample_iteration(pvnet_config_filename):
    """Tests iterating raw samples (dict of tensors) from PVNetDataset"""
    dataset = PVNetDataset(pvnet_config_filename)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=None,
        shuffle=False,
        num_workers=0,
    )

    raw_sample = next(iter(dataloader))

    # Assertions for the raw sample
    assert isinstance(
        raw_sample,
        dict,
    ), "Sample yielded by DataLoader with batch_size=None should be a dict"

    required_keys = [
        "nwp_ukv",
        "satellite",
        "generation_input",
        "generation_target",
        "solar_azimuth",
        "solar_elevation",
        "date_sin",
        "date_cos",
        "time_sin",
        "time_cos",
        "location_id",
    ]
    for key in required_keys:
        assert key in raw_sample, f"Raw Sample: Expected key '{key}' not found"

    # Type assertions
    assert isinstance(raw_sample["satellite"], torch.Tensor)
    assert isinstance(raw_sample["generation_input"], torch.Tensor)
    assert isinstance(raw_sample["generation_target"], torch.Tensor)
    assert isinstance(raw_sample["solar_azimuth"], torch.Tensor)
    assert isinstance(raw_sample["solar_elevation"], torch.Tensor)
    assert isinstance(raw_sample["nwp_ukv"], torch.Tensor)

    # Shape assertions
    assert raw_sample["satellite"].shape == (7, 1, 2, 2)
    assert raw_sample["nwp_ukv"].shape == (4, 1, 2, 2)
    assert raw_sample["generation_input"].shape == (3,)
    assert raw_sample["generation_target"].shape == (5,)

    # Solar position shapes - no batch dimension
    solar_config = dataset.config.solar_position
    expected_time_steps = (
        solar_config.interval_end_minutes - solar_config.interval_start_minutes
    ) // solar_config.time_resolution_minutes + 1
    assert raw_sample["solar_azimuth"].shape == (expected_time_steps,)
    assert raw_sample["solar_elevation"].shape == (expected_time_steps,)

    # Datetime encoding shapes - no batch dimension
    dt_config = dataset.config.datetime_encoding
    expected_dt_time_steps = (
        dt_config.interval_end_minutes - dt_config.interval_start_minutes
    ) // dt_config.time_resolution_minutes + 1
    for key in ("date_sin", "date_cos", "time_sin", "time_cos"):
        assert raw_sample[key].shape == (expected_dt_time_steps,)

    assert isinstance(raw_sample["location_id"], int | np.integer)


def test_pvnet_dataset_pickle(tmp_path, pvnet_config_filename):
    pickle_path = f"{tmp_path}.pkl"
    dataset = PVNetDataset(pvnet_config_filename)

    # Assert path is in pickle object
    dataset.presave_pickle(pickle_path)
    pickle_bytes = pickle.dumps(dataset)
    assert pickle_path.encode("utf-8") in pickle_bytes

    # Check we can reload the object
    _ = pickle.loads(pickle_bytes)  # noqa: S301

    # Check we can still pickle and unpickle if we don't presave
    dataset = PVNetDataset(pvnet_config_filename)
    pickle_bytes = pickle.dumps(dataset)
    _ = pickle.loads(pickle_bytes)  # noqa: S301


def test_pvnet_dataset_get_sample(pvnet_config_filename):
    dataset = PVNetDataset(
        pvnet_config_filename,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )
    # Test helper function get_sample to retrieve sample by t0 and location_id
    assert dataset.complete_generation
    t0 = dataset.valid_t0_times[0]
    location_id = next(iter(dataset.location_lookup))
    sample = dataset.get_sample(t0=t0, location_id=location_id)
    assert isinstance(sample, dict)

    # Check error raised if loc_id does not exist
    with pytest.raises(ValueError):
        sample = dataset.get_sample(t0=t0, location_id=400)

    # Check error raised if t0 does not exist
    with pytest.raises(ValueError):
        t0 = pd.Timestamp("2024-01-01 06:00")
        sample = dataset.get_sample(t0=t0, location_id=location_id)


def test_pvnet_dataset_sites_get_sample(pvnet_site_config_filename):
    dataset = PVNetDataset(
        pvnet_site_config_filename,
        time_periods=[
            ("2023-01-01 06:00", "2023-01-01 07:00"),
            ("2023-01-01 12:00", "2023-01-01 13:00"),
        ],
    )
    # Test helper function get_sample to retrieve sample by t0 and location_id
    assert not dataset.complete_generation
    t0 = dataset.valid_t0_and_location_ids["t0"].values[0]
    location_id = dataset.valid_t0_and_location_ids["location_id"].values[0]
    sample = dataset.get_sample(t0=t0, location_id=location_id)
    assert isinstance(sample, dict)

    # Check error raised if loc_id does not exist
    with pytest.raises(ValueError):
        sample = dataset.get_sample(t0=t0, location_id=400)

    # Check error raised if t0 does not exist
    with pytest.raises(ValueError):
        t0 = np.datetime64("2024-01-01 06:00")
        sample = dataset.get_sample(t0=t0, location_id=location_id)
