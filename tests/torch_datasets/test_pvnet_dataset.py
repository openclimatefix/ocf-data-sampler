import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.pvnet_dataset import (
    PVNetConcurrentDataset,
    PVNetDataset,
)
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)


def test_pvnet_dataset(pvnet_config_filename):
    dataset = PVNetDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317  # Quantity of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317 * 39

    sample = dataset[0]
    assert isinstance(sample, dict)

    # Specific keys should always be present
    required_keys = ["nwp", "satellite_actual", "generation", "t0", "t0_embedding"]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if dataset.config.input_data.solar_position is not None:
        # Test solar position keys are present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from config
        expected_time_steps = (
            dataset.config.input_data.solar_position.interval_end_minutes
            - dataset.config.input_data.solar_position.interval_start_minutes
        ) // dataset.config.input_data.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on config
        assert sample["solar_azimuth"].shape == (expected_time_steps,)
        assert sample["solar_elevation"].shape == (expected_time_steps,)
    else:
        # Assert that solar position keys are not present
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite_actual"].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["generation"].shape == (7,)
    # datetime encoding keys same shape as the generation
    for datetime_key in ["date_sin", "date_cos", "time_sin", "time_cos"]:
        assert sample[datetime_key].shape == (7,)
    # The config uses 3 periods each of which generates a sin and cos embedding
    assert sample["t0_embedding"].shape == (6,)


def test_pvnet_dataset_sites(pvnet_site_config_filename):
    dataset = PVNetDataset(pvnet_site_config_filename)

    assert len(dataset.locations) == 10
    # max possible t0s is 10 * 39 if full, so should be less than that
    assert len(dataset.valid_t0_and_location_ids) < 10 * 39

    sample = dataset[0]
    assert isinstance(sample, dict)

    # Specific keys should always be present
    required_keys = ["nwp", "satellite_actual", "generation", "t0"]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if dataset.config.input_data.solar_position is not None:
        # Test solar position keys are present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from config
        expected_time_steps = (
            dataset.config.input_data.solar_position.interval_end_minutes
            - dataset.config.input_data.solar_position.interval_start_minutes
        ) // dataset.config.input_data.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on config
        assert sample["solar_azimuth"].shape == (expected_time_steps,)
        assert sample["solar_elevation"].shape == (expected_time_steps,)
    else:
        # Assert that solar position keys are not present
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite_actual"].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["generation"].shape == (7,)


def test_pvnet_concurrent_dataset(pvnet_config_filename):
    # Create dataset object using limited set of GSPs
    dataset = PVNetConcurrentDataset(pvnet_config_filename)
    num_gsps = 317
    assert len(dataset.locations) == num_gsps  # Quantity of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 39

    sample = dataset[0]
    assert isinstance(sample, dict)

    required_keys = ["nwp", "satellite_actual", "generation"]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if dataset.config.input_data.solar_position is not None:
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        expected_time_steps = (
            dataset.config.input_data.solar_position.interval_end_minutes
            - dataset.config.input_data.solar_position.interval_start_minutes
        ) // dataset.config.input_data.solar_position.time_resolution_minutes + 1

        assert sample["solar_azimuth"].shape == (num_gsps, expected_time_steps)
        assert sample["solar_elevation"].shape == (num_gsps, expected_time_steps)
    else:
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Shape assertion checking
    assert sample["satellite_actual"].shape == (num_gsps, 7, 1, 2, 2)
    assert sample["nwp"]["ukv"]["nwp"].shape == (num_gsps, 4, 1, 2, 2)
    assert sample["generation"].shape == (num_gsps, 7)


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
        "nwp",
        "satellite_actual",
        "generation",
        "solar_azimuth",
        "solar_elevation",
        "location_id",
    ]
    for key in required_keys:
        assert key in raw_sample, f"Raw Sample: Expected key '{key}' not found"

    # Type assertions
    assert isinstance(raw_sample["satellite_actual"], torch.Tensor)
    assert isinstance(raw_sample["generation"], torch.Tensor)
    assert isinstance(raw_sample["solar_azimuth"], torch.Tensor)
    assert isinstance(raw_sample["solar_elevation"], torch.Tensor)
    assert isinstance(raw_sample["nwp"], dict)
    assert "ukv" in raw_sample["nwp"]
    assert isinstance(raw_sample["nwp"]["ukv"]["nwp"], torch.Tensor)
    assert isinstance(raw_sample["nwp"]["ukv"]["nwp_channel_names"], np.ndarray)

    # Shape assertions
    assert raw_sample["satellite_actual"].shape == (7, 1, 2, 2)
    assert raw_sample["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
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


def test_pvnet_dataset_batch_size_2(pvnet_config_filename):
    """Tests making batches from PVNetDataset"""
    dataset = PVNetDataset(pvnet_config_filename)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=stack_np_samples_into_batch,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    batch = batch_to_tensor(batch)
    batch = copy_batch_to_device(batch, torch.device("cpu"))

    # Assertions for the raw batch
    assert isinstance(batch, dict), "Sample yielded by DataLoader with batch_size=2 should be dict"

    required_keys = [
        "nwp",
        "satellite_actual",
        "generation",
        "solar_azimuth",
        "solar_elevation",
        "location_id",
        "t0",
    ]
    for key in required_keys:
        assert key in batch, f"Raw Sample: Expected key '{key}' not found"

    # Type assertions
    assert isinstance(batch["satellite_actual"], torch.Tensor)
    assert isinstance(batch["generation"], torch.Tensor)
    assert isinstance(batch["solar_azimuth"], torch.Tensor)
    assert isinstance(batch["solar_elevation"], torch.Tensor)
    assert isinstance(batch["nwp"], dict)
    assert "ukv" in batch["nwp"]
    assert isinstance(batch["nwp"]["ukv"]["nwp"], torch.Tensor)
    assert isinstance(batch["nwp"]["ukv"]["nwp_channel_names"], np.ndarray)
    assert isinstance(batch["t0"], torch.Tensor)

    # Shape assertions
    assert batch["satellite_actual"].shape == (2, 7, 1, 2, 2)
    assert batch["nwp"]["ukv"]["nwp"].shape == (2, 4, 1, 2, 2)
    assert batch["generation"].shape == (2, 7)
    assert batch["t0"].shape == (2,)
