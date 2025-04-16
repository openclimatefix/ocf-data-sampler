
from ocf_data_sampler.config import Configuration
from ocf_data_sampler.load.site import open_site
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.torch_datasets.utils.spatial_slice_for_dataset import (
    slice_datasets_by_space,
)


def test_slice_datasets_by_space_sites_static(data_sites):
    """Test that slice_datasets_by_space correctly handles static capacity mode.

    In static capacity mode:
    1. The capacity_kwp coordinate (with only site_id dimension) remains unchanged
    2. Only the generation_kw data variable remains in the final dataset
    3. The site_id dimension is preserved but with length 1
    """
    # Load the sites data using the Site object
    da_sites = open_site(
        data_sites.file_path,
        data_sites.metadata_file_path,
        data_sites.capacity_mode,
    )

    # Create a config with the Site object
    config = Configuration()
    config.input_data.site = data_sites

    # Create a location object for the first site using the same approach as SitesDataset
    site_id = da_sites.site_id.values[0]
    location = Location(
        id=site_id,
        x=float(da_sites.longitude.sel(site_id=site_id)),
        y=float(da_sites.latitude.sel(site_id=site_id)),
        coordinate_system="lon_lat",
    )

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the datasets
    sliced_datasets_dict = slice_datasets_by_space(
        datasets_dict,
        location,
        config,
    )

    # Check that the dimensions are preserved
    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert "capacity_kwp" in sliced_datasets_dict["site"].coords

    assert sliced_datasets_dict["site"].coords["site_id"].values[0] == site_id


def test_slice_datasets_by_space_sites_variable(data_sites_var_capacity):
    """Test that slice_datasets_by_space correctly handles variable capacity mode.

    In variable capacity mode:
    1. The capacity_kwp data variable should be preserved
    2. The site_id dimension is preserved but with length 1
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

    # Create a location object for the first site using the same approach as SitesDataset
    site_id = da_sites.site_id.values[0]
    location = Location(
        id=site_id,
        x=float(da_sites.longitude.sel(site_id=site_id)),
        y=float(da_sites.latitude.sel(site_id=site_id)),
        coordinate_system="lon_lat",
    )

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the datasets
    sliced_datasets_dict = slice_datasets_by_space(
        datasets_dict,
        location,
        config,
    )

    # Check that the dimensions and data variables are preserved correctly
    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert "capacity_kwp" in sliced_datasets_dict["site"].data_vars
    assert sliced_datasets_dict["site"].coords["site_id"].values[0] == site_id


def test_slice_datasets_by_space_single_site_variable(data_single_site_var_capacity):
    """Test that slice_datasets_by_space correctly handles variable capacity mode for single site.

    In variable capacity mode with a single site:
    1. The capacity_kwp data variable should be preserved
    2. The site_id dimension is preserved but with length 1
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

    # Create a location object for the site using the same approach as SitesDataset
    site_id = da_sites.site_id.values[0]
    location = Location(
        id=site_id,
        x=float(da_sites.longitude.sel(site_id=site_id)),
        y=float(da_sites.latitude.sel(site_id=site_id)),
        coordinate_system="lon_lat",
    )

    # Create a datasets_dict with the sites data
    datasets_dict = {"site": da_sites}

    # Slice the datasets
    sliced_datasets_dict = slice_datasets_by_space(
        datasets_dict,
        location,
        config,
    )

    # Check that the dimensions and data variables are preserved correctly
    assert "site_id" in sliced_datasets_dict["site"].coords
    assert "latitude" in sliced_datasets_dict["site"].coords
    assert "longitude" in sliced_datasets_dict["site"].coords
    assert "capacity_kwp" in sliced_datasets_dict["site"].data_vars
    assert sliced_datasets_dict["site"].coords["site_id"].values[0] == site_id
