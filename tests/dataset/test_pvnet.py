import pytest
import tempfile

from ocf_data_sampler.datasets.pvnet import PVNetDataset
from ocf_datapipes.config.load import load_yaml_configuration
from ocf_datapipes.config.save import save_yaml_configuration


@pytest.fixture()
def pvnet_config_filename(config_filename, nwp_ukv_zarr_path, uk_gsp_zarr_path, sat_zarr_path):

    # adjust config to point to the zarr file
    config = load_yaml_configuration(config_filename)
    config.input_data.nwp['ukv'].nwp_zarr_path = nwp_ukv_zarr_path
    config.input_data.satellite.satellite_zarr_path = sat_zarr_path
    config.input_data.gsp.gsp_zarr_path = uk_gsp_zarr_path

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/configuration.yaml"
        save_yaml_configuration(config, filename)
        yield filename


def test_pvnet(pvnet_config_filename):

    # Create dataset object
    dataset = PVNetDataset(pvnet_config_filename)

    # Print number of samples
    print(f"Found {len(dataset.valid_t0_times)} possible samples")

    idx = 0
    t_index, loc_index = dataset.index_pairs[idx]

    location = dataset.locations[loc_index]
    t0 = dataset.valid_t0_times[t_index]

    # Print coords
    print(t0)
    print(location)

    # Generate sample - no printing since its BIG
    _ = dataset[idx]
