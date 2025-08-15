"""Utility function for converting normalisation constants in the config to arrays."""

import numpy as np

from ocf_data_sampler.config import Configuration


def config_normalization_values_to_dicts(
    config: Configuration,
) -> tuple[dict[str, np.ndarray | dict[str, np.ndarray]]]:
    """Construct numpy arrays of mean and std values from the config normalisation constants.

    Args:
        config: Data configuration.

    Returns:
        Means dict
        Stds dict
    """
    means_dict = {}
    stds_dict = {}

    if config.input_data.nwp is not None:

        means_dict["nwp"] = {}
        stds_dict["nwp"] = {}

        for nwp_key in config.input_data.nwp:
            nwp_config = config.input_data.nwp[nwp_key]

            means_list = []
            stds_list = []

            for channel in list(nwp_config.channels):
                # These accumulated channels are diffed and renamed
                if channel in nwp_config.accum_channels:
                    channel =f"diff_{channel}"

                means_list.append(nwp_config.normalisation_constants[channel].mean)
                stds_list.append(nwp_config.normalisation_constants[channel].std)

            means_dict["nwp"][nwp_key] = np.array(means_list)[None, :, None, None]
            stds_dict["nwp"][nwp_key] = np.array(stds_list)[None, :, None, None]

    if config.input_data.satellite is not None:
        sat_config = config.input_data.satellite

        means_list = []
        stds_list = []

        for channel in list(config.input_data.satellite.channels):
            means_list.append(sat_config.normalisation_constants[channel].mean)
            stds_list.append(sat_config.normalisation_constants[channel].std)

        # Convert to array and expand dimensions so we can normalise the 4D sat and NWP sources
        means_dict["sat"] = np.array(means_list)[None, :, None, None]
        stds_dict["sat"] = np.array(stds_list)[None, :, None, None]

    return means_dict, stds_dict
