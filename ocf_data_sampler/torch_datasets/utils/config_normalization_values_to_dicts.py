"""Utility function for converting channel dictionaries to xarray DataArrays."""

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
) -> tuple[dict[str, xr.DataArray | dict[str, xr.DataArray]]]:
    """Construct DataArrays of mean and std values from the config normalisation constants.

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
            # Standardise and convert to NumpyBatch

            means_dict["nwp"][nwp_key] = channel_dict_to_dataarray(
                config.input_data.nwp[nwp_key].channel_means,
            )
            stds_dict["nwp"][nwp_key] = channel_dict_to_dataarray(
                config.input_data.nwp[nwp_key].channel_stds,
            )

    if config.input_data.satellite is not None:

        means_dict["sat"] = channel_dict_to_dataarray(config.input_data.satellite.channel_means)
        stds_dict["sat"] = channel_dict_to_dataarray(config.input_data.satellite.channel_stds)

    return means_dict, stds_dict
