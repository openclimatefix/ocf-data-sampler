import pandas as pd

from ocf_data_sampler.torch_datasets import SitesDataset
from ocf_data_sampler.torch_datasets.site import convert_from_dataset_to_dict_datasets

from xarray import Dataset


def test_site(site_config_filename):

    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    assert len(dataset) == 10 * 41
    # TODO check 41

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, Dataset)

    # Expected dimensions and data variables
    expected_dims = {'satellite__x_geostationary', 'sites__time_utc', 'nwp-ukv__target_time_utc',
                     'nwp-ukv__x_osgb', 'satellite__channel', 'satellite__y_geostationary',
                     'satellite__time_utc', 'nwp-ukv__channel', 'nwp-ukv__y_osgb'}
    expected_data_vars = {"nwp-ukv", "satellite", "sites"}

    # Check dimensions
    assert set(sample.dims) == expected_dims, f"Missing or extra dimensions: {set(sample.dims) ^ expected_dims}"
    # Check data variables
    assert set(sample.data_vars) == expected_data_vars, f"Missing or extra data variables: {set(sample.data_vars) ^ expected_data_vars}"

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite"].values.shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp-ukv"].values.shape == (4, 1, 2, 2)
    # 1.5 hours of 30 minute data (inclusive)
    assert sample["sites"].values.shape == (4,)

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

    print(sample.keys())

    for key in ["nwp", "satellite", "sites"]:
        assert key in sample
