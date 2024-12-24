from ocf_data_sampler.numpy_sample import GSPSampleKey, SatelliteSampleKey
from ocf_data_sampler.numpy_sample.collate import stack_np_examples_into_sample
from ocf_data_sampler.torch_datasets import PVNetUKRegionalDataset


def test_pvnet(pvnet_config_filename):

    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317 * 39

    # Generate 2 samples
    sample1 = dataset[0]
    sample2 = dataset[1]

    sample = stack_np_examples_into_sample([sample1, sample2])

    assert isinstance(sample, dict)
    assert "nwp" in sample
    assert isinstance(sample["nwp"], dict)
    assert "ukv" in sample["nwp"]
    assert GSPSampleKey.gsp in sample
    assert SatelliteSampleKey.satellite_actual in sample
