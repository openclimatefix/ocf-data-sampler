"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

import numpy as np

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.features.diff_channels import diff_channels
from ocf_data_sampler.select.dropout import apply_dropout


def config_normalization_values_to_dicts(
    config: PVNetDataConfig,
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

    if config.nwp is not None:

        means_dict["nwp"] = {}
        stds_dict["nwp"] = {}
        clip_min_dict["nwp"] = {}
        clip_max_dict["nwp"] = {}

        for nwp_key, nwp_config in config.nwp.items():

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

    if config.satellite is not None:

        means_list = []
        stds_list = []
        clip_min_list = []
        clip_max_list = []

        for channel in list(config.satellite.channels):
            norm_conf = config.satellite.normalisation_constants[channel]
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


def normalise_dataset_dicts(
    dataset_dict: SourceDict,
    mean_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    std_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    clip_min_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    clip_max_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> SourceDict:
    """Normalise the NWP, satellite, and generation data in-place.

    NWP and satellite are normalised using the per-channel mean/std/clip constants from config.
    Generation is normalised differently: `generation_mw` is rescaled to a capacity factor by
    dividing by `capacity_mwp`, which is time-varying, per-location data rather than a config
    constant - so it can't use the same clip/mean/std path. `capacity_mwp` itself is left
    unchanged, since it's exposed raw in the output sample.

    Args:
        dataset_dict: Dictionary of xarray datasets
        mean_dict: Means, as constructed by `config_normalization_values_to_dicts`
        std_dict: Standard deviations, as constructed by `config_normalization_values_to_dicts`
        clip_min_dict: Clip minimums, as constructed by `config_normalization_values_to_dicts`
        clip_max_dict: Clip maximums, as constructed by `config_normalization_values_to_dicts`
    """
    if "nwp" in dataset_dict:
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            channel_means = mean_dict["nwp"][nwp_key]
            channel_stds = std_dict["nwp"][nwp_key]
            channel_mins = clip_min_dict["nwp"][nwp_key]
            channel_maxs = clip_max_dict["nwp"][nwp_key]
            dataset_dict["nwp"][nwp_key].data = (
                (da_nwp.data.clip(channel_mins, channel_maxs) - channel_means)
                / channel_stds
            )

    if "sat" in dataset_dict:
        channel_means = mean_dict["sat"]
        channel_stds = std_dict["sat"]
        channel_mins = clip_min_dict["sat"]
        channel_maxs = clip_max_dict["sat"]
        dataset_dict["sat"].data = (
            (dataset_dict["sat"].data.clip(channel_mins, channel_maxs) - channel_means)
            / channel_stds
        )

    for key in ("generation_input", "generation_target"):
        if key not in dataset_dict:
            continue

        da = dataset_dict[key]
        gen_idx = list(da["gen_param"].values).index("generation_mw")
        cap_idx = list(da["gen_param"].values).index("capacity_mwp")

        generation_values = da.isel(gen_param=gen_idx).values
        capacity_values = da.isel(gen_param=cap_idx).values

        # capacity_mwp is time-varying (per timestep, per location) - normalise element-wise
        # rather than by a single scalar. Where capacity is 0 the ratio is undefined, so we
        # emit NaN rather than silently switching units (raw MW) or dividing by zero - dropout
        # fill (later in the pipeline) replaces it with a fixed value in normalised units.
        da.data[..., gen_idx] = np.divide(
            generation_values,
            capacity_values,
            out=np.full_like(generation_values, np.nan, dtype=float),
            where=capacity_values != 0,
        )

    return dataset_dict


def diff_nwp_data(dataset_dict: SourceDict, config: PVNetDataConfig) -> SourceDict:
    """Take the in-place diff of some channels of the NWP data.

    Args:
        dataset_dict: Dictionary of xarray datasets
        config: PVNetDataConfig object
    """
    if "nwp" in dataset_dict:
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            accum_channels = config.nwp[nwp_key].accum_channels
            if len(accum_channels)>0:
                # diff_channels() is an in-place operation and modifies the input
                dataset_dict["nwp"][nwp_key] = diff_channels(da_nwp, accum_channels)
    return dataset_dict


def apply_dropout_to_datasets(
    datasets_dict: SourceDict,
    t0: np.datetime64,
    config: PVNetDataConfig,
) -> None:
    """Apply dropout in-placeto the dictionary of input data sources around a given t0 time.

    Args:
        datasets_dict: Dictionary of the input data sources
        t0: The init-time
        config: PVNetDataConfig object.

    Returns:
        None. The input datasets_dict is modified in place.
    """
    if "sat" in datasets_dict:
        apply_dropout(
            datasets_dict["sat"],
            t0,
            dropout_timedeltas=minutes(config.satellite.dropout_timedeltas_minutes),
            dropout_frac=config.satellite.dropout_fraction,
        )

    if "generation_input" in datasets_dict:

        # capacity_mwp is dropped out along with generation_mw - if the input feed was stale for
        # this timestep, we didn't know the capacity at that point either.
        # generation_target is never dropped out - it's the prediction target, not an input.
        apply_dropout(
            datasets_dict["generation_input"],
            t0,
            dropout_timedeltas=minutes(config.generation.input.dropout_timedeltas_minutes),
            dropout_frac=config.generation.input.dropout_fraction,
        )

    return


def fill_nans_in_dataset_dicts(datasets_dict: SourceDict, config: PVNetDataConfig) -> SourceDict:
    """Fills all NaN values in the dataarrays in-place.

    Args:
        datasets_dict: Dictionary of the input data sources
        config: PVNetDataConfig object.
    """
    if config.generation is not None:
        for key, window_config in (
            ("generation_input", config.generation.input),
            ("generation_target", config.generation.target),
        ):
            if key in datasets_dict:
                datasets_dict[key] = fill_nans(
                    datasets_dict[key], window_config.dropout_fill_value,
                )

    if "sat" in datasets_dict:
        datasets_dict["sat"] = fill_nans(datasets_dict["sat"], config.satellite.dropout_fill_value)

    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.nwp.items():
            datasets_dict["nwp"][nwp_key] = fill_nans(
                datasets_dict["nwp"][nwp_key],
                nwp_config.dropout_fill_value,
            )

    return datasets_dict


def fill_nans(da: TArray, fill_value: float) -> TArray:
    """Fill NaNs in a DataArray in-place."""
    if np.isnan(da.data).any():
        da.data = np.nan_to_num(da.data, copy=True, nan=fill_value)
    return da


def preprocess_dataset_dict(
    dataset_dict: SourceDict,
    t0: np.datetime64,
    config: PVNetDataConfig,
    mean_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    std_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    clip_min_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
    clip_max_dict: dict[str, np.ndarray | dict[str, np.ndarray]],
) -> SourceDict:
    """Diff, normalise, dropout, and fill NaNs in the dictionary of input data sources.

    These steps are always applied in the order listed, since some steps depend on the output of
    previous steps. For example,
    - NWP channel differencing must be done before normalisation, since the diffed channels have
      different statistics
    - Dropout must be applied after normalisation, since the fill value is specified in normalised
      units
    - NaN filling must be done after dropout, since dropout introduces NaNs in the data

    Note: `dataset_dict` is expected to already be loaded - see `load_data_dict`.

    Args:
        dataset_dict: Dictionary of xarray datasets
        t0: The init-time
        config: PVNetDataConfig object
        mean_dict: Means, as constructed by `config_normalization_values_to_dicts`
        std_dict: Standard deviations, as constructed by `config_normalization_values_to_dicts`
        clip_min_dict: Clip minimums, as constructed by `config_normalization_values_to_dicts`
        clip_max_dict: Clip maximums, as constructed by `config_normalization_values_to_dicts`
    """
    dataset_dict = diff_nwp_data(dataset_dict, config)
    dataset_dict = normalise_dataset_dicts(
        dataset_dict, mean_dict, std_dict, clip_min_dict, clip_max_dict,
    )
    apply_dropout_to_datasets(dataset_dict, t0, config)
    dataset_dict = fill_nans_in_dataset_dicts(dataset_dict, config=config)
    return dataset_dict
