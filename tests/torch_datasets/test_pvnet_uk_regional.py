import pytest
import tempfile

from ocf_data_sampler.torch_datasets.pvnet_uk_regional import PVNetUKRegionalDataset
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_datapipes.batch import BatchKey, NWPBatchKey


@pytest.fixture()
def pvnet_config_filename(tmp_path, config_filename, nwp_ukv_zarr_path, uk_gsp_zarr_path, sat_zarr_path):

    # adjust config to point to the zarr file
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp['ukv'].nwp_zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.satellite_zarr_path = sat_zarr_path
    config.input_data.gsp.gsp_zarr_path = uk_gsp_zarr_path

    filename = f"{tmp_path}/configuration.yaml"
    save_yaml_configuration(config, filename)
    return filename


def test_pvnet(pvnet_config_filename):

    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317 # no of GSPs not including the National level
    # NB. I have not checked this value is in fact correct, but it does seem to stay constant
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317*39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    for key in [
        BatchKey.nwp, BatchKey.satellite_actual, BatchKey.gsp,
        BatchKey.gsp_solar_azimuth, BatchKey.gsp_solar_elevation,
    ]:
        assert key in sample
    
    for nwp_source in ["ukv"]:
        assert nwp_source in sample[BatchKey.nwp]

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample[BatchKey.satellite_actual].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample[BatchKey.nwp]["ukv"][NWPBatchKey.nwp].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample[BatchKey.gsp].shape == (7,)
    # Solar angles have same shape as GSP data
    assert sample[BatchKey.gsp_solar_azimuth].shape == (7,)
    assert sample[BatchKey.gsp_solar_elevation].shape == (7,)

def test_pvnet_no_gsp(pvnet_config_filename):

    # load config
    config = load_yaml_configuration(pvnet_config_filename)
    # remove gsp
    config.input_data.gsp.gsp_zarr_path = ''

    # save temp config file
    with tempfile.NamedTemporaryFile() as temp_config_file:
        save_yaml_configuration(config, temp_config_file.name)
        # Create dataset object
        dataset = PVNetUKRegionalDataset(temp_config_file.name)

        # Generate a sample
        _ = dataset[0]
