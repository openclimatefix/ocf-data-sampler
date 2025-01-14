import pytest
import tempfile

from ocf_data_sampler.torch_datasets import PVNetUKRegionalDataset
from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.numpy_sample import NWPSampleKey, GSPSampleKey, SatelliteSampleKey



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
        NWPSampleKey.nwp, SatelliteSampleKey.satellite_actual, GSPSampleKey.gsp,
        GSPSampleKey.solar_azimuth, GSPSampleKey.solar_elevation,
    ]:
        assert key in sample
    
    for nwp_source in ["ukv"]:
        assert nwp_source in sample[NWPSampleKey.nwp]

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample[NWPSampleKey.nwp]["ukv"][NWPSampleKey.nwp].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample[GSPSampleKey.gsp].shape == (7,)
    # Solar angles have same shape as GSP data
    assert sample[GSPSampleKey.solar_azimuth].shape == (7,)
    assert sample[GSPSampleKey.solar_elevation].shape == (7,)

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
