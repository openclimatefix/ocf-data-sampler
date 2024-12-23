from ocf_data_sampler.numpy_batch import GSPBatchKey, SatelliteBatchKey
from ocf_data_sampler.numpy_batch.collate import stack_np_examples_into_batch
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

    batch = stack_np_examples_into_batch([sample1, sample2])

    assert isinstance(batch, dict)
    assert "nwp" in batch
    assert isinstance(batch["nwp"], dict)
    assert "ukv" in batch["nwp"]
    assert GSPBatchKey.gsp in batch
    assert SatelliteBatchKey.satellite_actual in batch
