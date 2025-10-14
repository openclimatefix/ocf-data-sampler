import numpy as np
import pandas as pd

from ocf_data_sampler.torch_datasets.datasets.energy_forecast import EnergyForecastDataset
from ocf_data_sampler.torch_datasets.utils.time_slice_for_dataset import slice_datasets_by_time


def test_time_slice_for_dataset_site_dropout(site_config_filename):
    dataset = EnergyForecastDataset(site_config_filename)
    config = dataset.config

    # Set dropout
    config.input_data.site.dropout_timedeltas_minutes = [-30]
    config.input_data.site.dropout_fraction = 1.0

    sliced = slice_datasets_by_time(
        datasets_dict=dataset.datasets_dict,
        t0=pd.Timestamp("2023-01-01 12:00"),
        config=config,
    )

    site_dataset = sliced["site"]

    # For all 10 site IDs the second time step should be NaN due to dropout
    assert np.all(np.isnan(site_dataset[1, :]))
    # The last time step (after t0) should not be impacted by dropout
    assert np.all(~np.isnan(site_dataset[-1, :]))
