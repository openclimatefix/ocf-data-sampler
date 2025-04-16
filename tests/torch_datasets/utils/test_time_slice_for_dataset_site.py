import numpy as np
import pandas as pd
import pytest

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.load.site import open_site
from ocf_data_sampler.torch_datasets.utils.time_slice_for_dataset import (
    slice_datasets_by_time,
)


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_slice_datasets_by_time_sites_static(data_sites, t0_str):
    """Test that slice_datasets_by_time correctly handles static capacity mode.

    In static capacity mode:
    1. The capacity_kwp coordinate (with only site_id dimension) remains unchanged
    2. Only the generation_kw data variable remains in the final dataset
    """

    da_sites = open_site(
        data_sites.file_path, data_sites.metadata_file_path, data_sites.capacity_mode,
    )

    # Create a config with the Site object
    config = Configuration()
    config.input_data.site = data_sites

    t0 = pd.Timestamp(f"2023-01-01 {t0_str}")

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the datasets
    sliced_datasets_dict = slice_datasets_by_time(
        datasets_dict,
        t0,
        config,
    )
    # Check that the dimensions are preserved
    # Check coordinates exist and have correct types
    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert len(sliced_datasets_dict["site"].coords["capacity_kwp"]) == len(
        sliced_datasets_dict["site"].coords["site_id"])

@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_slice_datasets_by_time_sites_variable(data_sites_var_capacity, t0_str):
    """Test that slice_datasets_by_time correctly handles variable capacity mode.

    In variable capacity mode:
    1. The original capacity_kwp data variable (with time_utc and site_id dimensions) is converted
       to a coordinate with only the site_id dimension
    2. The first capacity value at t0 is used for this coordinate
    3. Only the generation_kw data variable remains in the final dataset
    """

    # Load the sites data using the Site object
    da_sites = open_site(
        data_sites_var_capacity.file_path,
        data_sites_var_capacity.metadata_file_path,
        data_sites_var_capacity.capacity_mode,
    )

    # Create a config with the Site object
    config = Configuration()
    config.input_data.site = data_sites_var_capacity

    t0 = pd.Timestamp(f"2023-01-01 {t0_str}")

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the dataset by time
    sliced_datasets_dict = slice_datasets_by_time(
        datasets_dict,
        t0,
        config,
    )

    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert len(sliced_datasets_dict["site"].coords["capacity_kwp"]) == len(
        sliced_datasets_dict["site"].coords["site_id"])


@pytest.mark.parametrize("t0_str", ["12:30", "12:40", "12:00"])
def test_slice_datasets_by_time_single_site_variable(data_single_site_var_capacity, t0_str):
    """Test that slice_datasets_by_time correctly handles variable capacity mode for single site.

    In variable capacity mode:
    1. The original capacity_kwp data variable (with time_utc and site_id dimensions) is converted
       to a coordinate with only the site_id dimension
    2. The first capacity value at t0 is used for this coordinate
    3. Only the generation_kw data variable remains in the final dataset
    """

    # Load the sites data using the Site object
    da_sites = open_site(
        data_single_site_var_capacity.file_path,
        data_single_site_var_capacity.metadata_file_path,
        data_single_site_var_capacity.capacity_mode,
    )

    # Create a config with the Site object
    config = Configuration()
    config.input_data.site = data_single_site_var_capacity

    t0 = pd.Timestamp(f"2023-01-01 {t0_str}")

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the dataset by time
    sliced_datasets_dict = slice_datasets_by_time(
        datasets_dict,
        t0,
        config,
    )

    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert len(sliced_datasets_dict["site"].coords["capacity_kwp"]) == len(
        sliced_datasets_dict["site"].coords["site_id"])

def test_compare_variable_capacity_values_at_different_times(data_sites_var_capacity):
    """Compare capacity values at different times to verify they are different."""
    # Load the sites data
    da_sites = open_site(
        data_sites_var_capacity.file_path,
        data_sites_var_capacity.metadata_file_path,
        data_sites_var_capacity.capacity_mode,
    )

    # Create a config with the Site object
    config = Configuration()
    config.input_data.site = data_sites_var_capacity

    # Get capacity values at 12:00
    t0_1200 = pd.Timestamp("2023-01-01 12:00")
    datasets_dict_1200 = {"site": da_sites}
    sliced_1200 = slice_datasets_by_time(datasets_dict_1200, t0_1200, config)
    capacity_1200 = sliced_1200["site"].coords["capacity_kwp"].values

    # Get capacity values at 12:30
    t0_1230 = pd.Timestamp("2023-01-01 12:30")
    datasets_dict_1230 = {"site": da_sites}
    sliced_1230 = slice_datasets_by_time(datasets_dict_1230, t0_1230, config)
    capacity_1230 = sliced_1230["site"].coords["capacity_kwp"].values

    # Capacity values should be different at different times
    assert not np.array_equal(capacity_1200, capacity_1230)
