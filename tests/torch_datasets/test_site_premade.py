import pandas as pd
import pytest
import tempfile

from ocf_data_sampler.torch_datasets import SitesDataset
from ocf_data_sampler.torch_datasets.site_premade import (
    SitesPreMadeSamplesDataset,
    convert_from_dataset_to_dict_datasets,
)
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_batch.nwp import NWPBatchKey
from ocf_data_sampler.numpy_batch.site import SiteBatchKey
from ocf_data_sampler.numpy_batch.satellite import SatelliteBatchKey
from xarray import Dataset


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


def test_convert_netcdf_to_numpy_sample(site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    # Generate two samples
    sample_xr = dataset[0]

    dataset = SitesPreMadeSamplesDataset(".", site_config_filename)

    sample = dataset.convert_netcdf_to_numpy_sample(sample_xr)

    assert isinstance(sample, dict)

    for key in [
        NWPBatchKey.nwp,
        SatelliteBatchKey.satellite_actual,
        SiteBatchKey.generation,
        # SiteBatchKey.site_solar_azimuth,
        # SiteBatchKey.site_solar_elevation,
    ]:
        assert key in sample


def test_site(site_config_filename):

    # Create dataset object
    dataset = SitesDataset(site_config_filename)

    # Generate two samples
    sample1 = dataset[0]
    sample2 = dataset[1]

    # make temp directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # save to temporary directory
        sample1.to_netcdf(f"{tmpdirname}/sample1.nc")
        sample2.to_netcdf(f"{tmpdirname}/sample2.nc")

        dataset = SitesPreMadeSamplesDataset(tmpdirname, site_config_filename)

        assert len(dataset) == 2
        sample = dataset[0]

        assert isinstance(sample, dict)

        for key in [
            NWPBatchKey.nwp,
            SatelliteBatchKey.satellite_actual,
            SiteBatchKey.generation,
            # SiteBatchKey.site_solar_azimuth,
            # SiteBatchKey.site_solar_elevation,
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
        # assert sample[SiteBatchKey.site_solar_azimuth].shape == (4,)
        # assert sample[SiteBatchKey.site_solar_elevation].shape == (4,)
