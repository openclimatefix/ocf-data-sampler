import pandas as pd
import pytest

from ocf_data_sampler.torch_datasets import SitesDataset
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.numpy_batch.site import SiteBatchKey
from ocf_data_sampler.numpy_batch.satellite import SatelliteBatchKey
from xarray import Dataset


@pytest.fixture()
def site_config_filename(tmp_path, config_filename, nwp_ukv_zarr_path, sat_zarr_path, data_sites):

    # adjust config to point to the zarr file
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp["ukv"].zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.zarr_path = sat_zarr_path
    config.input_data.site = data_sites
    config.input_data.gsp = None

    filename = f"{tmp_path}/configuration.yaml"
    save_yaml_configuration(config, filename)
    return filename


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
