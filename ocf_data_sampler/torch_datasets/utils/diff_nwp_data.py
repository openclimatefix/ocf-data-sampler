"""Take the in-place diff of some channels of the NWP data."""

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.diff_channels import diff_channels


def diff_nwp_data(dataset_dict: dict, config: Configuration) -> dict:
    """Take the in-place diff of some channels of the NWP data.

    Args:
        dataset_dict: Dictionary of xarray datasets
        config: Configuration object
    """
    if "nwp" in dataset_dict:
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            accum_channels = config.input_data.nwp[nwp_key].accum_channels
            if len(accum_channels)>0:
                # diff_channels() is an in-place operation and modifies the input
                dataset_dict["nwp"][nwp_key] = diff_channels(da_nwp, accum_channels)
    return dataset_dict
