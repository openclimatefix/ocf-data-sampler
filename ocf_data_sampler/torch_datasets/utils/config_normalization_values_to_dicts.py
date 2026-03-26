"""Utility function for converting normalisation constants in the config to arrays."""

import numpy as np

from ocf_data_sampler.config import Configuration


def config_normalization_values_to_dicts(
    config: Configuration,
) -> tuple[dict[str, np.ndarray | dict[str, np.ndarray]]]:
    """Construct numpy arrays of mean, std, and clip values from the config normalisation constants.

    Args:
        config: Data configuration.

    Returns:
        Means dict
        Stds dict
        Clip min dict
        Clip max dict
    """
    means_dict = {}
    stds_dict = {}
    clip_min_dict = {}
    clip_max_dict = {}

    if config.input_data.nwp is not None:

        means_dict["nwp"] = {}
        stds_dict["nwp"] = {}
        clip_min_dict["nwp"] = {}
        clip_max_dict["nwp"] = {}

        for nwp_key in config.input_data.nwp:
            nwp_config = config.input_data.nwp[nwp_key]

            means_list = []
            stds_list = []
            clip_min_list = []
            clip_max_list = []

            for channel in list(nwp_config.channels):
                # These accumulated channels are diffed and renamed
                if channel in nwp_config.accum_channels:
                    channel =f"diff_{channel}"

                norm_conf = nwp_config.normalisation_constants[channel]

                means_list.append(norm_conf.mean)
                stds_list.append(norm_conf.std)
                clip_min_list.append(-np.inf if norm_conf.clip_min is None else norm_conf.clip_min)
                clip_max_list.append(np.inf if norm_conf.clip_max is None else norm_conf.clip_max)

            means_dict["nwp"][nwp_key] = np.array(means_list)[None, :, None, None]
            stds_dict["nwp"][nwp_key] = np.array(stds_list)[None, :, None, None]
            clip_min_dict["nwp"][nwp_key] = np.array(clip_min_list)[None, :, None, None]
            clip_max_dict["nwp"][nwp_key] = np.array(clip_max_list)[None, :, None, None]

    if config.input_data.satellite is not None:
        sat_config = config.input_data.satellite

        means_list = []
        stds_list = []
        clip_min_list = []
        clip_max_list = []

        for channel in list(sat_config.channels):
            norm_conf = sat_config.normalisation_constants[channel]
            means_list.append(norm_conf.mean)
            stds_list.append(norm_conf.std)
            clip_min_list.append(-np.inf if norm_conf.clip_min is None else norm_conf.clip_min)
            clip_max_list.append(np.inf if norm_conf.clip_max is None else norm_conf.clip_max)

        # Convert to array and expand dimensions so we can normalise the 4D sat and NWP sources
        means_dict["sat"] = np.array(means_list)[None, :, None, None]
        stds_dict["sat"] = np.array(stds_list)[None, :, None, None]
        clip_min_dict["sat"] = np.array(clip_min_list)[None, :, None, None]
        clip_max_dict["sat"] = np.array(clip_max_list)[None, :, None, None]

    return means_dict, stds_dict, clip_min_dict, clip_max_dict
