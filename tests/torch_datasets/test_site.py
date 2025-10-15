import pytest
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.torch_datasets.pvnet_dataset import (
    PVNetConcurrentDataset,
    PVNetDataset,
)


def test_site(site_config_filename):
    dataset = PVNetDataset(site_config_filename)
    # (10 sites * 24 valid t0s per site = 240)
    assert len(dataset) == 240

    # Generate a sample
    sample = dataset[0]
    assert isinstance(sample, dict)

    # Expected keys
    expected_keys = {
        "date_cos",
        "date_sin",
        "time_cos",
        "time_sin",
        "solar_azimuth",
        "solar_elevation",
        "satellite_x_geostationary",
        "satellite_time_utc",
        "satellite_y_geostationary",
        "satellite_actual",
        "site",
        "site_id",
        "site_time_utc",
        "site_capacity_kwp",
        "nwp",
        "t0",
    }
    assert set(sample.keys()) == expected_keys, (
        f"Missing or extra dimensions: {set(sample.keys()) ^ expected_keys}"
    )

    # Check shapes based on config intervals and image sizes
    # Satellite: (0 - (-30)) / 5 + 1 = 7 time steps; 2 channels; 24x24 pixels
    assert sample["satellite_actual"].shape == (7, 2, 24, 24)
    # NWP-UKV: (480 - (-60)) / 60 + 1 = 10 time steps; 1 channel; 24x24 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (10, 1, 24, 24)
    # Site: (60 - (-30)) / 30 + 1 = 4 time steps
    assert sample["site"].shape == (4,)


def test_site_time_filter_start(site_config_filename):
    dataset = PVNetDataset(site_config_filename, start_time="2024-01-01")
    assert len(dataset) == 0


def test_site_time_filter_end(site_config_filename):
    dataset = PVNetDataset(site_config_filename, end_time="2000-01-01")
    assert len(dataset) == 0


def test_site_dataset_with_dataloader(sites_dataset) -> None:
    if len(sites_dataset) == 0:
        pytest.skip("Skipping test as dataset is empty.")

    dataloader = DataLoader(
        sites_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=None,
    )

    try:
        individual_sample = next(iter(dataloader))
    except StopIteration:
        pytest.skip("Skipping test as dataloader is empty.")

    assert isinstance(individual_sample, dict)

    # Check shapes based on config intervals and image sizes
    assert individual_sample["satellite_actual"].shape == (7, 2, 24, 24)
    assert individual_sample["nwp"]["ukv"]["nwp"].shape == (10, 1, 24, 24)
    assert individual_sample["site"].shape == (4,)


def test_solar_position_decoupling_site(tmp_path, site_config_filename):
    """Test that solar position calculations are properly decoupled from data sources."""
    config = load_yaml_configuration(site_config_filename)
    config_without_solar = config.model_copy(deep=True)
    config_without_solar.input_data.solar_position = None

    # Version with explicit solar position config
    config_with_solar = config.model_copy(deep=True)
    config_with_solar.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    # Save both testing configs
    config_without_solar_path = tmp_path / "site_config_without_solar.yaml"
    config_with_solar_path = tmp_path / "site_config_with_solar.yaml"
    save_yaml_configuration(config_without_solar, config_without_solar_path)
    save_yaml_configuration(config_with_solar, config_with_solar_path)

    # Create datasets and generate samples
    dataset_without_solar = PVNetDataset(config_without_solar_path)
    dataset_with_solar = PVNetDataset(config_with_solar_path)
    sample_without_solar = dataset_without_solar[0]
    sample_with_solar = dataset_with_solar[0]

    # Assert solar position keys presence/absence
    for key in ["solar_azimuth", "solar_elevation"]:
        assert key not in sample_without_solar, f"Solar key {key} should not be in sample"
        assert key in sample_with_solar, f"Solar key {key} should be in sample"

def test_site_concurrent_dataset(site_config_filename):
    dataset = PVNetConcurrentDataset(site_config_filename)
    number_of_sites = len(dataset.locations)
    assert number_of_sites == 10  # Quantity of sites

    # t0 times repeated per site
    assert len(dataset.valid_t0_times) == 240
    assert len(dataset) == 240 // number_of_sites

    sample = dataset[0]
    assert isinstance(sample, dict)

    required_keys = ["nwp", "satellite_actual", "site"]
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

        assert sample["solar_azimuth"].shape == (number_of_sites, expected_time_steps)
        assert sample["solar_elevation"].shape == (number_of_sites, expected_time_steps)
    else:
        for key in solar_keys:
            assert key not in sample, f"Solar position key {key} should not be present"

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Shape assertion checking
    assert sample["satellite_actual"].shape == (number_of_sites, 7, 2, 24, 24)
    assert sample["nwp"]["ukv"]["nwp"].shape == (number_of_sites, 10, 1, 24, 24)
    assert sample["site"].shape == (number_of_sites, 4)
