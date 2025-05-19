from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from ocf_data_sampler.torch_datasets.utils.time_slice_for_dataset import slice_datasets_by_time
import numpy as np
import pandas as pd


def test_time_slice_for_dataset_site_dropout(tmp_path, site_config_filename):
    # Create dataset object
    dataset = SitesDataset(site_config_filename)
    datasets_dict = dataset.datasets_dict
    config = dataset.config

    # set dropout
    config.input_data.site.dropout_timedeltas_minutes = [-30]
    config.input_data.site.dropout_fraction = 1.0

    sliced_datasets_dict = slice_datasets_by_time(
        datasets_dict=datasets_dict, t0=pd.Timestamp("2023-01-01 12:00"), config=config
    )

    site_dataset = sliced_datasets_dict["site"]

    # for all 10 site ids the second element should nan due to dropout
    assert np.all(np.isnan(site_dataset[1, :]))
    # the last element which is after t0 should not be impacted by dropout
    assert np.all(~np.isnan(site_dataset[-1, :]))
