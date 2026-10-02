"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

from collections.abc import Sequence

import numpy as np

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import NormalisationValues, PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import (
    NormalisationArrays,
    SourceDict,
    SourceNormalisationArrays,
)
from ocf_data_sampler.features.diff_channels import diff_channels
from ocf_data_sampler.features.normalisation import clip_and_standardise
from ocf_data_sampler.select.dropout import apply_dropout


def _build_source_normalisation_arrays(
    values: Sequence[NormalisationValues]
) -> SourceNormalisationArrays:
    """Build parameters broadcastable over (time, channel, y, x)."""
    def channel_array(channel_values: list[float]) -> np.ndarray:
        return np.array(channel_values, dtype=np.float32)[None, :, None, None]

    return {
        "mean": channel_array([v.mean for v in values]),
        "std": channel_array([v.std for v in values]),
        "clip_min": channel_array([-np.inf if v.clip_min is None else v.clip_min for v in values]),
        "clip_max": channel_array([np.inf if v.clip_max is None else v.clip_max for v in values]),
    }


def build_normalisation_arrays(config: PVNetDataConfig) -> NormalisationArrays:
    """Build normalisation arrays in configured channel order for each source."""
    normalisation_arrays: NormalisationArrays = {}
    if config.nwp is not None:
        normalisation_arrays["nwp"] = {}
        for nwp_source, nwp_config in config.nwp.items():
            channels = [
                f"diff_{channel}" if channel in nwp_config.accum_channels else channel
                for channel in nwp_config.channels
            ]
            normalisation_arrays["nwp"][nwp_source] = _build_source_normalisation_arrays(
                [nwp_config.normalisation_constants[channel] for channel in channels]
            )
    if config.satellite is not None:
        sat_config = config.satellite
        normalisation_arrays["sat"] = _build_source_normalisation_arrays(
            [sat_config.normalisation_constants[channel] for channel in sat_config.channels]
        )
    return normalisation_arrays


def normalise_dataset_dicts(
    dataset_dict: SourceDict,
    normalisation_arrays: NormalisationArrays,
) -> SourceDict:
    """Normalise NWP, satellite, and generation data in-place."""
    if "nwp" in dataset_dict:
        for nwp_source, da in dataset_dict["nwp"].items():
            da.data = clip_and_standardise(da.data, **normalisation_arrays["nwp"][nwp_source])

    if "sat" in dataset_dict:
        da = dataset_dict["sat"]
        da.data = clip_and_standardise(da.data, **normalisation_arrays["sat"])

    for key in ("generation_input", "generation_target"):
        if key in dataset_dict:
            dataset_dict[key] = normalise_generation_by_capacity(dataset_dict[key])

    return dataset_dict


def normalise_generation_by_capacity(da: TArray) -> TArray:
    """Rescale `generation_mw` to a capacity factor, leaving `capacity_mwp` unchanged.

    Zero capacity means no plant, so the capacity factor is taken as 0 rather than the undefined
    0/0. Emitting NaN instead would route it through the dropout fill, which signals missing data
    rather than a known-zero output.

    Args:
        da: Generation DataArray-like with a `gen_param` dimension
    """
    gen_params = list(da["gen_param"].values)
    gen_idx = gen_params.index("generation_mw")
    cap_idx = gen_params.index("capacity_mwp")

    generation_values = da.isel(gen_param=gen_idx).values
    capacity_values = da.isel(gen_param=cap_idx).values

    normalised = np.divide(
        generation_values,
        capacity_values,
        out=np.zeros_like(generation_values, dtype=float),
        where=capacity_values != 0,
    )

    new_data = da.data.copy()
    index = [slice(None)] * new_data.ndim
    index[da.dims.index("gen_param")] = gen_idx
    new_data[tuple(index)] = normalised
    da.data = new_data

    return da


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
    """Apply dropout in-place to the dictionary of input data sources around a given t0 time.

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

    # generation_target is never dropped out since it's the prediction target
    if "generation_input" in datasets_dict:

        # Note: capacity_mwp is dropped out along with generation_mw
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
    normalisation_arrays: NormalisationArrays,
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
        normalisation_arrays: Precomputed arrays from `build_normalisation_arrays`
    """
    dataset_dict = diff_nwp_data(dataset_dict, config)
    dataset_dict = normalise_dataset_dicts(dataset_dict, normalisation_arrays)
    apply_dropout_to_datasets(dataset_dict, t0, config)
    dataset_dict = fill_nans_in_dataset_dicts(dataset_dict, config=config)
    return dataset_dict
