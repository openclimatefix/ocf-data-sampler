"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

from collections.abc import Sequence

import numpy as np

from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import Channel, PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import (
    NormalisationArrays,
    SourceDict,
)
from ocf_data_sampler.features.diff_channels import diff_channels
from ocf_data_sampler.features.normalisation import clip_and_standardise
from ocf_data_sampler.select.dropout import apply_dropout, choose_delay


def _build_source_normalisation_arrays(values: Sequence[Channel]) -> NormalisationArrays:
    """Build parameters broadcastable over (time, channel, x, y)."""
    def channel_array(channel_values: list[float]) -> np.ndarray:
        return np.array(channel_values, dtype=np.float32)[None, :, None, None]

    return NormalisationArrays(
        mean=channel_array([v.mean for v in values]),
        std=channel_array([v.std for v in values]),
        clip_min=channel_array([-np.inf if v.clip_min is None else v.clip_min for v in values]),
        clip_max=channel_array([np.inf if v.clip_max is None else v.clip_max for v in values]),
    )


def build_normalisation_arrays(config: PVNetDataConfig) -> SourceDict[NormalisationArrays]:
    """Build normalisation arrays in configured channel order for each source."""
    normalisation_arrays: SourceDict[NormalisationArrays] = {}
    if config.nwp is not None:
        normalisation_arrays["nwp"] = {}
        for nwp_source, nwp_config in config.nwp.items():
            normalisation_arrays["nwp"][nwp_source] = _build_source_normalisation_arrays(
                nwp_config.channels,
            )
    if config.satellite is not None:
        normalisation_arrays["sat"] = _build_source_normalisation_arrays(config.satellite.channels)
    return normalisation_arrays


def normalise_dataset_dicts(
    dataset_dict: SourceDict,
    normalisation_arrays: SourceDict[NormalisationArrays],
) -> None:
    """Normalise NWP, satellite, and generation data in-place."""
    if "nwp" in dataset_dict:
        for nwp_source, da in dataset_dict["nwp"].items():
            params = normalisation_arrays["nwp"][nwp_source]
            da.data = clip_and_standardise(
                da.data,
                clip_min=params.clip_min,
                clip_max=params.clip_max,
                mean=params.mean,
                std=params.std,
            )

    if "sat" in dataset_dict:
        da = dataset_dict["sat"]
        params = normalisation_arrays["sat"]
        da.data = clip_and_standardise(
            da.data,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            mean=params.mean,
            std=params.std,
        )

    for key in ("generation_input", "generation_target"):
        if key in dataset_dict:
            normalise_generation_by_capacity(dataset_dict[key])


def normalise_generation_by_capacity(da: TArray) -> None:
    """Rescale `generation_mw` to a capacity factor, leaving `capacity_mwp` unchanged.

    Replaces `.data` on the input DataArray without modifying shared source values.

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
        out=np.zeros_like(generation_values),
        where=capacity_values != 0,
    )

    new_data = da.data.copy()
    index = [slice(None)] * new_data.ndim
    index[da.dims.index("gen_param")] = gen_idx
    new_data[tuple(index)] = normalised
    da.data = new_data


def diff_nwp_data(dataset_dict: SourceDict, config: PVNetDataConfig) -> None:
    """Difference accumulated NWP channels, updating the input dictionary in-place.

    Args:
        dataset_dict: Dictionary of xarray datasets
        config: PVNetDataConfig object
    """
    if "nwp" in dataset_dict:
        for nwp_key, da_nwp in dataset_dict["nwp"].items():
            accum_channels = [
                channel.name for channel in config.nwp[nwp_key].channels if channel.accumulated
            ]
            if len(accum_channels)>0:
                dataset_dict["nwp"][nwp_key] = diff_channels(da_nwp, accum_channels)


def apply_dropout_to_datasets(
    datasets_dict: SourceDict,
    t0: np.datetime64,
    config: PVNetDataConfig,
) -> None:
    """Apply dropout to input DataArrays in-place around a given t0 time.

    Args:
        datasets_dict: Dictionary of the input data sources
        t0: The init-time
        config: PVNetDataConfig object.

    """
    sat_conf = config.satellite
    if sat_conf is not None and sat_conf.dropout_minute_probabilities is not None:
        apply_dropout(
            datasets_dict["sat"],
            t0,
            dropout_delay=choose_delay(sat_conf.dropout_minute_probabilities),
        )

    # generation_target is never dropped out since it's the prediction target
    gen_conf = config.generation
    if (
        gen_conf is not None
        and gen_conf.input is not None
        and gen_conf.input.dropout_minute_probabilities is not None
    ):

        # Note: capacity_mwp is dropped out along with generation_mw
        apply_dropout(
            datasets_dict["generation_input"],
            t0,
            dropout_delay=choose_delay(gen_conf.input.dropout_minute_probabilities),
        )

    return


def fill_nans_in_dataset_dicts(datasets_dict: SourceDict, config: PVNetDataConfig) -> None:
    """Fill NaNs in-place in input sources where a fill value is configured.

    Args:
        datasets_dict: Dictionary of the input data sources
        config: PVNetDataConfig object.
    """
    if (
        "generation_input" in datasets_dict
        and config.generation.input.fill_value is not None
    ):
        fill_nans(datasets_dict["generation_input"], config.generation.input.fill_value)

    if "sat" in datasets_dict and config.satellite.fill_value is not None:
        fill_nans(datasets_dict["sat"], config.satellite.fill_value)

    if "nwp" in datasets_dict:
        for nwp_source, da in datasets_dict["nwp"].items():
            fill_value = config.nwp[nwp_source].fill_value
            if fill_value is not None:
                fill_nans(da, fill_value)



def fill_nans(da: TArray, fill_value: float) -> None:
    """Fill NaNs in a DataArray in-place."""
    if np.isnan(da.data).any():
        da.data = np.nan_to_num(da.data, copy=True, nan=fill_value)


def preprocess_dataset_dict(
    dataset_dict: SourceDict,
    t0: np.datetime64,
    config: PVNetDataConfig,
    normalisation_arrays: SourceDict[NormalisationArrays],
) -> None:
    """Diff, normalise, dropout, and fill NaNs in the dictionary of input data sources.

    These steps are always applied in the order listed, since some steps depend on the output of
    previous steps. For example,
    - NWP channel differencing must be done before normalisation, since the diffed channels have
      different statistics
    - Dropout must be applied after normalisation, since the fill value is specified in normalised
      units
    - NaN filling must be done after dropout, since dropout introduces NaNs in the data

    Mutates the supplied dictionary without modifying the original NumPy arrays.

    Note: `dataset_dict` is expected to already be loaded - see `materialise_data`.

    Args:
        dataset_dict: Dictionary of xarray datasets
        t0: The init-time
        config: PVNetDataConfig object
        normalisation_arrays: Precomputed arrays from `build_normalisation_arrays`
    """
    diff_nwp_data(dataset_dict, config)
    normalise_dataset_dicts(dataset_dict, normalisation_arrays)
    apply_dropout_to_datasets(dataset_dict, t0, config)
    fill_nans_in_dataset_dicts(dataset_dict, config=config)
