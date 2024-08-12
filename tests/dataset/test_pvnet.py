import pytest

from ocf_data_sampler.datasets.pvnet import PVNetDataset
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.save import save_yaml_configuration
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
    dataset = PVNetDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317
    assert len(dataset.valid_t0_times) == 39 # NB. I have not checked this value is in fact correct
    assert len(dataset) == 317*39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    for key in [BatchKey.nwp, BatchKey.satellite_actual, BatchKey.gsp]:
        assert key in sample
    
    for nwp_source in ["ukv"]:
        assert nwp_source in sample[BatchKey.nwp]
    
    assert sample[BatchKey.satellite_actual].shape == (7, 1, 2, 2)
    assert sample[BatchKey.nwp]["ukv"][NWPBatchKey.nwp].shape == (4, 1, 2, 2)
    assert sample[BatchKey.gsp].shape == (7,)


    
