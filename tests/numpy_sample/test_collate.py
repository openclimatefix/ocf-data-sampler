from ocf_data_sampler.numpy_sample.collate import stack_np_samples_into_batch
from ocf_data_sampler.torch_datasets.datasets.energy_forecast import EnergyForecastDataset


def test_stack_np_samples_into_batch(pvnet_config_filename):

    # Create dataset object - generate two samples
    dataset = EnergyForecastDataset(pvnet_config_filename)
    batch = stack_np_samples_into_batch([dataset[0], dataset[1]])

    assert isinstance(batch, dict)
    assert "nwp" in batch
    assert isinstance(batch["nwp"], dict)
    assert "ukv" in batch["nwp"]

    for key in ("gsp", "satellite_actual", "t0"):
        assert key in batch
