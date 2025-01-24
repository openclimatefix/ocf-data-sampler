import pandas as pd
import numpy as np
from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset, convert_from_dataset_to_dict_datasets, coarsen_data
from xarray import Dataset, DataArray
import xarray as xr

from torch.utils.data import DataLoader


def test_site(site_config_filename):

    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    assert len(dataset) == 10 * 41
    # TODO check 41

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, Dataset)

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
        "site__solar_azimuth",
        "site__solar_elevation",
        "site__date_cos",
        "site__time_cos",
        "site__time_sin",
        "site__date_sin",
    }

    expected_data_vars = {"nwp-ukv", "satellite", "site"}


    sample.to_netcdf("sample.nc")
    sample = xr.open_dataset("sample.nc")

    # Check dimensions
    assert (
        set(sample.dims) == expected_dims
    ), f"Missing or extra dimensions: {set(sample.dims) ^ expected_dims}"
    # Check data variables
    assert (
        set(sample.data_vars) == expected_data_vars
    ), f"Missing or extra data variables: {set(sample.data_vars) ^ expected_data_vars}"

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


def test_site_get_sample(site_config_filename):

    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    assert len(dataset) == 410
    sample = dataset.get_sample(t0=pd.Timestamp("2023-01-01 12:00"), site_id=1)


def test_convert_from_dataset_to_dict_datasets(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    # Generate two samples
    sample_xr = dataset[0]

    sample = convert_from_dataset_to_dict_datasets(sample_xr)

    assert isinstance(sample, dict)

    for key in ["nwp", "satellite", "site"]:
        assert key in sample


def test_site_dataset_with_dataloader(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    expected_coods = {
        "site__solar_azimuth",
        "site__solar_elevation",
        "site__date_cos",
        "site__time_cos",
        "site__time_sin",
        "site__date_sin",
    }

    sample = dataset[0]
    for key in expected_coods:
        assert key in sample

    dataloader_kwargs = dict(
        shuffle=False,
        batch_size=None,
        sampler=None,
        batch_sampler=None,
        num_workers=1,
        collate_fn=None,
        pin_memory=False,  # Only using CPU to prepare samples so pinning is not beneficial
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=1,
        persistent_workers=False,  # Not needed since we only enter the dataloader loop once
    )

    dataloader = DataLoader(dataset, collate_fn=None, batch_size=None)

    for i, sample in zip(range(1), dataloader):

        # check that expected_dims is in the sample
        for key in expected_coods:
            assert key in sample


def test_process_and_combine_site_sample_dict(site_config_filename):
    # Load config
    # config = load_yaml_configuration(pvnet_config_filename)
    site_ds = SitesDataset(site_config_filename)
    # Specify minimal structure for testing
    raw_nwp_values = np.random.rand(4, 1, 2, 2)  # Single channel
    fake_site_values = np.random.rand(197)
    site_dict = {
        "nwp": {
            "ukv": DataArray(
                raw_nwp_values,
                dims=["time_utc", "channel", "y", "x"],
                coords={
                    "time_utc": pd.date_range("2024-01-01 00:00", periods=4, freq="h"),
                    "channel": ["dswrf"],  # Single channel
                },
            )
        },
        "site": DataArray(
            fake_site_values,
            dims=["time_utc"],
            coords={
                    "time_utc": pd.date_range("2024-01-01 00:00", periods=197, freq="15min"),
                    "capacity_kwp": 1000,
                    "site_id": 1,
                    "longitude": -3.5,
                    "latitude": 51.5
                }
        )
    }
    print(f"Input site_dict: {site_dict}")

    # Call function
    result = site_ds.process_and_combine_site_sample_dict(site_dict)

    # Assert to validate output structure
    assert isinstance(result, Dataset), "Result should be an xarray.Dataset"
    assert len(result.data_vars) > 0, "Dataset should contain data variables"

    # Validate variable via assertion and shape of such
    expected_variables = ["nwp-ukv", "site"]
    for expected_variable in expected_variables:
        assert expected_variable in result.data_vars, f"Expected variable '{expected_variable}' not found"
    
    nwp_result = result["nwp-ukv"]
    assert nwp_result.shape == (4, 1, 2, 2), f"Unexpected shape for nwp-ukv : {nwp_result.shape}"
    site_result = result["site"]
    assert site_result.shape == (197,), f"Unexpected shape for site: {site_result.shape}"


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
