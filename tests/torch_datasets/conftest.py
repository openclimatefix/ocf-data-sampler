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

