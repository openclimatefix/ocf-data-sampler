import pytest
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.torch_datasets.datasets.site import (
    SitesDataset,
    coarsen_data,
)


def test_site(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    # (10 sites * 24 valid t0s per site = 240)
    assert len(dataset) == 240

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    # # Expected dimensions and data variables
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

    # Check keys
    assert set(sample.keys()) == expected_keys, (
        f"Missing or extra dimensions: {set(sample.keys()) ^ expected_keys}"
    )
    # check the shape of the data is correct based on new config intervals and image sizes
    # Satellite: (0 - (-30)) / 5 + 1 = 7 time steps; 2 channels (IR_016, VIS006); 24x24 pixels
    assert sample["satellite_actual"].shape == (7, 2, 24, 24)
    # NWP-UKV: (480 - (-60)) / 60 + 1 = 10 time steps; 1 channel (t); 24x24 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (10, 1, 24, 24)
    # Site: (60 - (-30)) / 30 + 1 = 4 time steps (from site_config_filename interval)
    assert sample["site"].shape == (4,)


def test_site_time_filter_start(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename, start_time="2024-01-01")

    assert len(dataset) == 0


def test_site_time_filter_end(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename, end_time="2000-01-01")

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
        return


    assert isinstance(individual_sample, dict)

    # check the shape of the data is correct based on new config intervals and image sizes
    # Satellite: (0 - (-30)) / 5 + 1 = 7 time steps; 2 channels (IR_016, VIS006); 24x24 pixels
    assert individual_sample["satellite_actual"].shape == (7, 2, 24, 24)
    # NWP-UKV: (480 - (-60)) / 60 + 1 = 10 time steps; 1 channel (t); 24x24 pixels
    assert individual_sample["nwp"]["ukv"]["nwp"].shape == (10, 1, 24, 24)
    # Site: (60 - (-30)) / 30 + 1 = 4 time steps (from site_config_filename interval)
    assert individual_sample["site"].shape == (4,)



def test_potentially_coarsen(ds_nwp_ecmwf):
    """Test potentially_coarsen function with ECMWF_UK data."""
    nwp_data = ds_nwp_ecmwf
    assert nwp_data.ECMWF_UK.shape[3:] == (15, 12)  # Check initial shape (lon, lat)

    data = coarsen_data(xr_data=nwp_data, coarsen_to_deg=2)
    assert data.ECMWF_UK.shape[3:] == (8, 6)  # Coarsen to every 2 degrees

    data = coarsen_data(xr_data=nwp_data, coarsen_to_deg=3)
    assert data.ECMWF_UK.shape[3:] == (5, 4)  # Coarsen to every 3 degrees

    data = coarsen_data(xr_data=nwp_data, coarsen_to_deg=1)
    assert data.ECMWF_UK.shape[3:] == (15, 12)  # No coarsening (same shape)


def test_solar_position_decoupling_site(tmp_path, site_config_filename):
    """Test that solar position calculations are properly decoupled from data sources."""

    config = load_yaml_configuration(site_config_filename)
    config_without_solar = config.model_copy(deep=True)
    config_without_solar.input_data.solar_position = None

    # Create version with explicit solar position configuration
    config_with_solar = config.model_copy(deep=True)
    config_with_solar.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    # Save both testing configurations
    config_without_solar_path = tmp_path / "site_config_without_solar.yaml"
    config_with_solar_path = tmp_path / "site_config_with_solar.yaml"
    save_yaml_configuration(config_without_solar, config_without_solar_path)
    save_yaml_configuration(config_with_solar, config_with_solar_path)

    # Create datasets with both configurations
    dataset_without_solar = SitesDataset(config_without_solar_path)
    dataset_with_solar = SitesDataset(config_with_solar_path)

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
