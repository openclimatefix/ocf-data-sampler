import dask.array
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import (
    PVNetUKConcurrentDataset,
    PVNetUKRegionalDataset,
    compute,
)


def test_compute():
    """Test compute function with dask array"""
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))

    # Create a nested dictionary with dask array
    lazy_data_dict = {
        "array1": da_dask,
        "nested": {
            "array2": da_dask,
        },
    }

    computed_data_dict = compute(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(computed_data_dict["array1"].data, np.ndarray)
    assert isinstance(computed_data_dict["nested"]["array2"].data, np.ndarray)


def test_pvnet_uk_regional_dataset(pvnet_config_filename):
    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317  # Number of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317 * 39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    # These keys should always be present
    required_keys = ["nwp", "satellite_actual", "gsp"]
    for key in required_keys:
        assert key in sample

    solar_keys = ["solar_azimuth", "solar_elevation"]
    if dataset.config.input_data.solar_position is not None:
        # Test that solar position keys are present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from configuration
        expected_time_steps = (
            dataset.config.input_data.solar_position.interval_end_minutes
            - dataset.config.input_data.solar_position.interval_start_minutes
        ) // dataset.config.input_data.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on config
        assert sample["solar_azimuth"].shape == (expected_time_steps,)
        assert sample["solar_elevation"].shape == (expected_time_steps,)
    else:
        # Test that solar position keys are not present
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
    assert sample["gsp"].shape == (7,)


def test_pvnet_uk_regional_dataset_limit_gsp_ids(pvnet_config_filename):
    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename, gsp_ids=[1, 2, 3])

    assert len(dataset.locations) == 3  # Number of regional GSPs
    assert len(dataset.datasets_dict["gsp"].gsp_id) == 3


def test_pvnet_no_gsp(tmp_path, pvnet_config_filename):
    # Create new config without GSP inputs
    config = load_yaml_configuration(pvnet_config_filename)
    config.input_data.gsp.zarr_path = ""
    new_config_path = tmp_path / "pvnet_config_no_gsp.yaml"
    save_yaml_configuration(config, new_config_path)

    # Create dataset object
    dataset = PVNetUKRegionalDataset(new_config_path)

    # Generate a sample
    _ = dataset[0]


def test_pvnet_uk_concurrent_dataset(pvnet_config_filename):
    # Create dataset object using a limited set of GSPs for test
    gsp_ids = [1, 2, 3]
    num_gsps = len(gsp_ids)

    dataset = PVNetUKConcurrentDataset(pvnet_config_filename, gsp_ids=gsp_ids)

    assert len(dataset.locations) == num_gsps  # Number of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    # These keys should always be present
    required_keys = ["nwp", "satellite_actual", "gsp"]
    for key in required_keys:
        assert key in sample

    # Check if solar position is configured in the dataset
    solar_keys = ["solar_azimuth", "solar_elevation"]
    if dataset.config.input_data.solar_position is not None:
        # Solar position keys should be present when configured
        for key in solar_keys:
            assert key in sample, f"Solar position key {key} should be present in sample"

        # Get expected time steps from configuration
        expected_time_steps = (
            dataset.config.input_data.solar_position.interval_end_minutes
            - dataset.config.input_data.solar_position.interval_start_minutes
        ) // dataset.config.input_data.solar_position.time_resolution_minutes + 1

        # Test solar angle shapes based on configuration
        assert sample["solar_azimuth"].shape == (num_gsps, expected_time_steps)
        assert sample["solar_elevation"].shape == (num_gsps, expected_time_steps)
    else:
        # Solar position keys should not be present when not configured
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite_actual"].shape == (num_gsps, 7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (num_gsps, 4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["gsp"].shape == (num_gsps, 7)


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

    # Save both testing configurations
    config_without_solar_path = tmp_path / "config_without_solar.yaml"
    config_with_solar_path = tmp_path / "config_with_solar.yaml"
    save_yaml_configuration(config_without_solar, config_without_solar_path)
    save_yaml_configuration(config_with_solar, config_with_solar_path)

    # Create datasets with both configurations
    dataset_without_solar = PVNetUKRegionalDataset(config_without_solar_path, gsp_ids=[1])
    dataset_with_solar = PVNetUKRegionalDataset(config_with_solar_path, gsp_ids=[1])

    # Generate samples
    sample_without_solar = dataset_without_solar[0]
    sample_with_solar = dataset_with_solar[0]

    # Assert solar position keys are only in sample specifically with solar configuration
    solar_keys = ["solar_azimuth", "solar_elevation"]

    # Sample without solar config should not have solar position data
    for key in solar_keys:
        assert key not in sample_without_solar, f"Solar key {key} should not be in sample"

    # Sample with solar config should have solar position data
    for key in solar_keys:
        assert key in sample_with_solar, f"Solar key {key} should be in sample"


def test_pvnet_uk_regional_dataset_raw_sample_iteration(pvnet_config_filename):
    """
    Tests iterating raw samples (dict of tensors) from PVNetUKRegionalDataset
    """

    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        collate_fn=None,
        shuffle=False,
        num_workers=0,
    )

    raw_sample = next(iter(dataloader))

    # Assertions for the raw sample
    assert isinstance(raw_sample, dict), \
        "Sample yielded by DataLoader with batch_size=None should be a dict"

    # Check for expected keys directly
    required_keys = ["nwp", "satellite_actual", "gsp", "solar_azimuth", "solar_elevation", "gsp_id"]
    for key in required_keys:
        assert key in raw_sample, f"Raw Sample: Expected key '{key}' not found"

    # Check types are primarily torch.Tensor
    assert isinstance(raw_sample["satellite_actual"], torch.Tensor)
    assert isinstance(raw_sample["gsp"], torch.Tensor)
    assert isinstance(raw_sample["solar_azimuth"], torch.Tensor)
    assert isinstance(raw_sample["solar_elevation"], torch.Tensor)
    assert isinstance(raw_sample["nwp"], dict)
    assert "ukv" in raw_sample["nwp"]
    assert isinstance(raw_sample["nwp"]["ukv"]["nwp"], torch.Tensor)
    assert isinstance(raw_sample["nwp"]["ukv"]["nwp_channel_names"], np.ndarray)

    # Check shapes
    assert raw_sample["satellite_actual"].shape == (7, 1, 2, 2)
    assert raw_sample["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    assert raw_sample["gsp"].shape == (7,)

    # Check solar position shapes (no batch dimension)
    solar_config = dataset.config.input_data.solar_position
    expected_time_steps = (
        solar_config.interval_end_minutes
        - solar_config.interval_start_minutes
    ) // solar_config.time_resolution_minutes + 1
    assert raw_sample["solar_azimuth"].shape == (expected_time_steps,)
    assert raw_sample["solar_elevation"].shape == (expected_time_steps,)

    assert isinstance(raw_sample["gsp_id"], int | np.integer)
