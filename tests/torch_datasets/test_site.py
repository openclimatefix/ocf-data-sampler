import pandas as pd
import pytest

from ocf_data_sampler.torch_datasets.site import SitesDataset
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.numpy_batch.site import SiteBatchKey
from ocf_data_sampler.numpy_batch.satellite import SatelliteBatchKey


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

    assert isinstance(sample, dict)

    for key in [
        NWPBatchKey.nwp,
        SatelliteBatchKey.satellite_actual,
        SiteBatchKey.generation,
        SiteBatchKey.site_solar_azimuth,
        SiteBatchKey.site_solar_elevation,
    ]:
        assert key in sample

    for nwp_source in ["ukv"]:
        assert nwp_source in sample[NWPBatchKey.nwp]

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample[SatelliteBatchKey.satellite_actual].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample[NWPBatchKey.nwp]["ukv"][NWPBatchKey.nwp].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample[SiteBatchKey.generation].shape == (4,)
    # Solar angles have same shape as GSP data
    assert sample[SiteBatchKey.site_solar_azimuth].shape == (4,)
    assert sample[SiteBatchKey.site_solar_elevation].shape == (4,)


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
