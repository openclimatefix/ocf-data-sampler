"""Utility function for converting channel dictionaries to xarray DataArrays."""

import numpy as np
import xarray as xr

from ocf_data_sampler.config import Configuration


def channel_dict_to_dataarray(channel_dict: dict[str, float]) -> xr.DataArray:
    """Converts a dictionary of channel values to a DataArray.

    Args:
        channel_dict: Dictionary mapping channel names (str) to their values (float).

    Returns:
        xr.DataArray: A 1D DataArray with channels as coordinates.
    """
    return xr.DataArray(
        list(channel_dict.values()),
        coords={"channel": list(channel_dict.keys())},
    )

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

            means_dict["nwp"][nwp_key] = np.array(means_list)
            stds_dict["nwp"][nwp_key] = np.array(stds_list)

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
