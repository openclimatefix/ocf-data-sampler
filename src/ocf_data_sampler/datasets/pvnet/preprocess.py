"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

import numpy as np

from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import Configuration
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.features.diff_channels import diff_channels


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


def diff_nwp_data(dataset_dict: SourceDict, config: Configuration) -> SourceDict:
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


def fill_nans_in_dataset_dicts(datasets_dict: SourceDict, config: Configuration) -> SourceDict:
    """Fills all NaN values in the dataarrays in-place.

    Args:
        datasets_dict: Dictionary of the input data sources
        config: Configuration object.
    """
    conf_in = config.input_data
    if "generation" in datasets_dict:
        datasets_dict["generation"] = fill_nans(
            datasets_dict["generation"],
            conf_in.generation.dropout_value,
        )

    if "sat" in datasets_dict:
        datasets_dict["sat"] = fill_nans(datasets_dict["sat"], conf_in.satellite.dropout_value)

    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.input_data.nwp.items():
            datasets_dict["nwp"][nwp_key] = fill_nans(
                datasets_dict["nwp"][nwp_key],
                nwp_config.dropout_value,
            )

    return datasets_dict


def fill_nans(da: TArray, fill_value: float) -> TArray:
    """Fill NaNs in a DataArray in-place."""
    if np.isnan(da.data).any():
        da.data = np.nan_to_num(da.data, copy=True, nan=fill_value)
    return da
