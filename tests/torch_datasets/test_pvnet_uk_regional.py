import pytest
import tempfile

from ocf_data_sampler.torch_datasets import PVNetUKRegionalDataset
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_batch import NWPBatchKey, GSPBatchKey, SatelliteBatchKey



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
        NWPBatchKey.nwp, SatelliteBatchKey.satellite_actual, GSPBatchKey.gsp,
        GSPBatchKey.solar_azimuth, GSPBatchKey.solar_elevation,
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
    assert sample[GSPBatchKey.gsp].shape == (7,)
    # Solar angles have same shape as GSP data
    assert sample[GSPBatchKey.solar_azimuth].shape == (7,)
    assert sample[GSPBatchKey.solar_elevation].shape == (7,)

def test_pvnet_no_gsp(pvnet_config_filename):

    # load config
    config = load_yaml_configuration(pvnet_config_filename)
    # remove gsp
    config.input_data.gsp.zarr_path = ''

    # save temp config file
    with tempfile.NamedTemporaryFile() as temp_config_file:
        save_yaml_configuration(config, temp_config_file.name)
        # Create dataset object
        dataset = PVNetUKRegionalDataset(temp_config_file.name)

        # Generate a sample
        _ = dataset[0]
