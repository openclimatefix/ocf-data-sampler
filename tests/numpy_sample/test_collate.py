from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset


def test_stack_np_samples_into_batch(pvnet_config_filename):
    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    # Generate 2 samples
    sample1 = dataset[0]
    sample2 = dataset[1]

    batch = stack_np_samples_into_batch([sample1, sample2])

    assert isinstance(batch, dict)
    assert "nwp" in batch
    assert isinstance(batch["nwp"], dict)
    assert "ukv" in batch["nwp"]
    assert "gsp" in batch
    assert "satellite_actual" in batch
    assert "t0" in batch
