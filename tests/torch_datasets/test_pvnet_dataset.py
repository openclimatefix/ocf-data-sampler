import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.torch_datasets.pvnet_dataset import (
    PVNetConcurrentDataset,
    PVNetDataset,
)


def _pvnet_dataset_sample_check(sample, config, batch_dim = None):
    """Helper function to verify samples"""

    if batch_dim is None:
        batch_dim = ()

    assert isinstance(sample, dict)

    # Specific keys should always be present
    required_keys = ["nwp_ukv", "satellite", "generation", "t0", "t0_embedding"]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if config.input_data.solar_position is not None:
        # Test solar position keys are present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from config
        expected_time_steps = (
            config.input_data.solar_position.interval_end_minutes
            - config.input_data.solar_position.interval_start_minutes
        ) // config.input_data.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on config
        assert sample["solar_azimuth"].shape == (*batch_dim, expected_time_steps)
        assert sample["solar_elevation"].shape == (*batch_dim, expected_time_steps)
    else:
        # Assert that solar position keys are not present
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite"].shape == (*batch_dim, 7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp_ukv"].shape == (*batch_dim, 4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["generation"].shape == (*batch_dim, 7)
    # Datetime encoding keys same shape as the generation
    for datetime_key in ["date_sin", "date_cos", "time_sin", "time_cos"]:
        assert sample[datetime_key].shape == (*batch_dim, 7)
    # The config uses 3 periods each of which generates a sin and cos embedding
    assert sample["t0_embedding"].shape == (*batch_dim, 6)



def test_pvnet_dataset(pvnet_config_filename):
    dataset = PVNetDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317  # Quantity of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317 * 39

    sample = dataset[0]

    _pvnet_dataset_sample_check(sample, dataset.config)


def test_pvnet_dataset_sites(pvnet_site_config_filename):
    dataset = PVNetDataset(pvnet_site_config_filename)

    assert len(dataset.locations) == 10
    # max possible t0s is 10 * 39 if full, so should be less than that
    assert len(dataset.valid_t0_and_location_ids) < 10 * 39

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
    num_gsps = 317
    assert len(dataset.locations) == num_gsps  # Quantity of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 39

    sample = dataset[0]
    _pvnet_dataset_sample_check(sample, dataset.config, (num_gsps,))


def test_solar_position_decoupling(tmp_path, pvnet_config_filename):
    """Test that solar position calculations are properly decoupled from data sources."""
    config = load_yaml_configuration(pvnet_config_filename)
    config_without_solar = config.model_copy(deep=True)
    config_without_solar.input_data.solar_position = None

    # Create version with explicit solar position configuration
    config_with_solar = config.model_copy(deep=True)
    config_with_solar.input_data.solar_position = SolarPosition(
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
        "generation",
        "solar_azimuth",
        "solar_elevation",
        "location_id",
    ]
    for key in required_keys:
        assert key in raw_sample, f"Raw Sample: Expected key '{key}' not found"

    # Type assertions
    assert isinstance(raw_sample["satellite"], torch.Tensor)
    assert isinstance(raw_sample["generation"], torch.Tensor)
    assert isinstance(raw_sample["solar_azimuth"], torch.Tensor)
    assert isinstance(raw_sample["solar_elevation"], torch.Tensor)
    assert isinstance(raw_sample["nwp_ukv"], torch.Tensor)

    # Shape assertions
    assert raw_sample["satellite"].shape == (7, 1, 2, 2)
    assert raw_sample["nwp_ukv"].shape == (4, 1, 2, 2)
    assert raw_sample["generation"].shape == (7,)

    # Solar position shapes - no batch dimension
    solar_config = dataset.config.input_data.solar_position
    expected_time_steps = (
        solar_config.interval_end_minutes - solar_config.interval_start_minutes
    ) // solar_config.time_resolution_minutes + 1
    assert raw_sample["solar_azimuth"].shape == (expected_time_steps,)
    assert raw_sample["solar_elevation"].shape == (expected_time_steps,)

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
