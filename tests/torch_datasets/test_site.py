import numpy as np
import pandas as pd
import pytest
import xarray as xr
from torch.utils.data import DataLoader

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import SolarPosition
from ocf_data_sampler.torch_datasets.datasets.site import (
    SitesDataset,
    coarsen_data,
    convert_from_dataset_to_dict_datasets,
    convert_netcdf_to_numpy_sample,
)


def test_site(tmp_path, site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    assert len(dataset) == 10 * 41
    # TODO check 41

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, xr.Dataset)

    # Expected dimensions and data variables
    expected_dims = {
        "satellite__x_geostationary",
        "site__time_utc",
        "nwp-ukv__target_time_utc",
        "nwp-ukv__x_osgb",
        "satellite__channel",
        "satellite__y_geostationary",
        "satellite__time_utc",
        "nwp-ukv__channel",
        "nwp-ukv__y_osgb",
    }

    expected_coords_subset = {
        "site__date_cos",
        "site__time_cos",
        "site__time_sin",
        "site__date_sin",
        "solar_azimuth",
        "solar_elevation"
    }

    expected_data_vars = {"nwp-ukv", "satellite", "site"}

    sample.to_netcdf(f"{tmp_path}/sample.nc", mode="w", engine="h5netcdf")
    sample = xr.open_dataset(f"{tmp_path}/sample.nc")

    # Check dimensions
    assert set(sample.dims) == expected_dims, (
        f"Missing or extra dimensions: {set(sample.dims) ^ expected_dims}"
    )
    # Check data variables
    assert set(sample.data_vars) == expected_data_vars, (
        f"Missing or extra data variables: {set(sample.data_vars) ^ expected_data_vars}"
    )

    print(sample.coords, "Sample coords")
    for coords in expected_coords_subset:
        assert coords in sample.coords

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite"].values.shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp-ukv"].values.shape == (4, 1, 2, 2)
    # 1.5 hours of 30 minute data (inclusive)
    assert sample["site"].values.shape == (4,)


def test_site_time_filter_start(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename, start_time="2024-01-01")

    assert len(dataset) == 0


def test_site_time_filter_end(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename, end_time="2000-01-01")

    assert len(dataset) == 0


def test_convert_from_dataset_to_dict_datasets(sites_dataset):
    # Generate sample
    sample_xr = sites_dataset[0]

    sample = convert_from_dataset_to_dict_datasets(sample_xr)

    assert isinstance(sample, dict)

    for key in ["nwp", "satellite", "site"]:
        assert key in sample


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
        individual_xr_sample = next(iter(dataloader))
    except StopIteration:
        pytest.skip("Skipping test as dataloader is empty.")
        return

    assert isinstance(individual_xr_sample, xr.Dataset)

    expected_data_vars = {"nwp-ukv", "satellite", "site"}
    assert set(individual_xr_sample.data_vars) == expected_data_vars

    assert individual_xr_sample["satellite"].values.shape == (7, 1, 2, 2)
    assert individual_xr_sample["nwp-ukv"].values.shape == (4, 1, 2, 2)
    assert individual_xr_sample["site"].values.shape == (4,)

    expected_coords_subset = {
        "site__date_cos",
        "site__time_cos",
        "site__time_sin",
        "site__date_sin",
        "solar_azimuth",
        "solar_elevation"
    }
    for coord_key in expected_coords_subset:
        assert coord_key in individual_xr_sample.coords


def test_process_and_combine_site_sample_dict(sites_dataset) -> None:
    # Specify minimal structure for testing
    raw_nwp_values = np.random.rand(4, 1, 2, 2)  # Single channel
    number_of_site_values = 4
    fake_site_values = np.random.rand(number_of_site_values)
    site_dict = {
        "nwp": {
            "ukv": xr.DataArray(
                raw_nwp_values,
                dims=["time_utc", "channel", "y", "x"],
                coords={
                    "time_utc": pd.date_range("2024-01-01 00:00", periods=4, freq="h"),
                    "channel": list(sites_dataset.config.input_data.nwp["ukv"].channels),
                },
            ),
        },
        "site": xr.DataArray(
            fake_site_values,
            dims=["time_utc"],
            coords={
                "time_utc": pd.date_range("2024-01-01 00:00", periods=number_of_site_values, freq="15min"),
                "capacity_kwp": 1000,
                "site_id": 1,
                "longitude": -3.5,
                "latitude": 51.5,
            },
        ),
    }

    t0 = pd.Timestamp("2024-01-01 00:00")

    # Call function
    result = sites_dataset.process_and_combine_site_sample_dict(site_dict, t0)

    # Assert to validate output structure
    assert isinstance(result, xr.Dataset), "Result should be an xarray.Dataset"
    assert len(result.data_vars) > 0, "Dataset should contain data variables"

    # Validate variable via assertion and shape of such
    expected_variables = ["nwp-ukv", "site"]
    for expected_variable in expected_variables:
        assert expected_variable in result.data_vars, (
            f"Expected variable '{expected_variable}' not found"
        )

    nwp_result = result["nwp-ukv"]
    assert nwp_result.shape == (4, 1, 2, 2), f"Unexpected shape for nwp-ukv : {nwp_result.shape}"
    site_result = result["site"]
    assert site_result.shape == (number_of_site_values,), f"Unexpected shape for site: {site_result.shape}"


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
        assert key not in sample_without_solar.coords, f"Solar key {key} should not be in sample"

    # Sample with solar config should have solar position data
    for key in solar_keys:
        assert key in sample_with_solar.coords, f"Solar key {key} should be in sample"


def test_convert_from_dataset_to_dict_solar_handling(tmp_path, site_config_filename):
    """Test that function handles solar position coordinates correctly."""

    config = load_yaml_configuration(site_config_filename)
    config.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    config_with_solar_path = tmp_path / "site_config_with_solar_for_dict.yaml"
    save_yaml_configuration(config, config_with_solar_path)

    # Create dataset and obtain sample with solar
    dataset_with_solar = SitesDataset(config_with_solar_path)
    sample_with_solar = dataset_with_solar[0]

    # Verify solar position data exists in original sample
    solar_keys = ["solar_azimuth", "solar_elevation"]
    for key in solar_keys:
        assert key in sample_with_solar.coords, f"Solar key {key} not found in original sample"

    # Conversion and subsequent verification
    converted_dict = convert_from_dataset_to_dict_datasets(sample_with_solar)
    assert isinstance(converted_dict, dict)
    assert "site" in converted_dict


def test_convert_netcdf_to_numpy_solar_handling(tmp_path, site_config_filename):
    """Test that convert_netcdf_to_numpy_sample handles solar position data correctly."""

    config = load_yaml_configuration(site_config_filename)
    config.input_data.solar_position = SolarPosition(
        time_resolution_minutes=30,
        interval_start_minutes=-30,
        interval_end_minutes=60,
    )

    config_with_solar_path = tmp_path / "site_config_with_solar_for_numpy.yaml"
    save_yaml_configuration(config, config_with_solar_path)

    # Create dataset and obtain sample with solar
    dataset_with_solar = SitesDataset(config_with_solar_path)
    sample_with_solar = dataset_with_solar[0]

    # Save to netCDF and load back
    netcdf_path = f"{tmp_path}/sample_with_solar.nc"
    sample_with_solar.to_netcdf(netcdf_path, mode="w", engine="h5netcdf")
    loaded_sample = xr.open_dataset(netcdf_path)

    # Verify solar position data exists in sample
    solar_keys = ["solar_azimuth", "solar_elevation"]
    for key in solar_keys:
        assert key in loaded_sample.coords, f"Solar key {key} not found in loaded netCDF"

    # Conversion and subsequent assertion
    numpy_sample = convert_netcdf_to_numpy_sample(loaded_sample)
    assert isinstance(numpy_sample, dict)

    # Explicitly verify what is in sample
    assert "nwp" in numpy_sample
    assert "satellite_actual" in numpy_sample or "sat" in numpy_sample
    assert "site" in numpy_sample

    # Assert solar position values exist in numpy sample
    for key in solar_keys:
        assert key in numpy_sample, f"Solar key {key} not found in numpy sample"
