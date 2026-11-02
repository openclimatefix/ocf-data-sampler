"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import Channel, PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import (
    NormalisationArrays,
    SourceDict,
)
from ocf_data_sampler.select.dropout import apply_dropout, choose_delay
from ocf_data_sampler.transforms.diff_channels import diff_channels_along_step
from ocf_data_sampler.transforms.missing_values import validate_and_fill_nans
from ocf_data_sampler.transforms.normalisation import (
    clip_and_standardise,
    normalise_generation_by_capacity,
)


def _build_source_normalisation_arrays(channels: Sequence[Channel]) -> NormalisationArrays:
    """Build parameters broadcastable over (time, channel, x, y)."""
    def channel_array(channel_values: list[float]) -> NDArray[np.float32]:
        return np.array(channel_values, dtype=np.float32)[None, :, None, None]

    return NormalisationArrays(
        mean=channel_array([c.mean for c in channels]),
        std=channel_array([c.std for c in channels]),
        clip_min=channel_array([-np.inf if c.clip_min is None else c.clip_min for c in channels]),
        clip_max=channel_array([np.inf if c.clip_max is None else c.clip_max for c in channels]),
    )


def build_normalisation_arrays(config: PVNetDataConfig) -> SourceDict[NormalisationArrays]:
    """Build normalisation arrays in configured channel order for each source."""
    normalisation_arrays: SourceDict[NormalisationArrays] = {}
    if config.nwp is not None:
        normalisation_arrays["nwp"] = {}
        for nwp_name, nwp_config in config.nwp.items():
            normalisation_arrays["nwp"][nwp_name] = _build_source_normalisation_arrays(
                nwp_config.channels,
            )
    if config.satellite is not None:
        normalisation_arrays["sat"] = _build_source_normalisation_arrays(config.satellite.channels)
    return normalisation_arrays


def normalise_source_data(
    source_data: SourceDict[TArray],
    normalisation_arrays: SourceDict[NormalisationArrays],
) -> None:
    """Normalise NWP, satellite, and generation data in-place."""
    if "nwp" in source_data:
        for nwp_name, da in source_data["nwp"].items():
            params = normalisation_arrays["nwp"][nwp_name]
            clip_and_standardise(
                da,
                clip_min=params.clip_min,
                clip_max=params.clip_max,
                mean=params.mean,
                std=params.std,
            )

    if "sat" in source_data:
        da = source_data["sat"]
        params = normalisation_arrays["sat"]
        clip_and_standardise(
            da,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            mean=params.mean,
            std=params.std,
        )

    for key in ("generation_input", "generation_target"):
        if key in source_data:
            normalise_generation_by_capacity(source_data[key])


def diff_accumulated_nwp_channels(source_data: SourceDict[TArray], config: PVNetDataConfig) -> None:
    """Difference accumulated NWP channels, updating the input dictionary in-place.

    Args:
        source_data: Source arrays containing the NWP data to difference
        config: PVNetDataConfig object
    """
    if "nwp" in source_data:
        for nwp_name, da_nwp in source_data["nwp"].items():
            accum_channels = [
                channel.name for channel in config.nwp[nwp_name].channels if channel.accumulated
            ]
            if len(accum_channels)>0:
                source_data["nwp"][nwp_name] = diff_channels_along_step(da_nwp, accum_channels)


def apply_source_dropout(
    source_data: SourceDict[TArray],
    t0: np.datetime64,
    config: PVNetDataConfig,
) -> None:
    """Apply dropout to input DataArrays in-place around a given t0 time.

    Args:
        source_data: Input source arrays
        t0: The init-time
        config: PVNetDataConfig object
    """
    sat_config = config.satellite
    if sat_config is not None and sat_config.dropout_minute_probabilities is not None:
        apply_dropout(
            source_data["sat"],
            t0,
            dropout_delay=choose_delay(sat_config.dropout_minute_probabilities),
        )

    # generation_target is never dropped out since it's the prediction target
    gen_config = config.generation
    if (
        gen_config is not None
        and gen_config.input is not None
        and gen_config.input.dropout_minute_probabilities is not None
    ):

        # Note: capacity_mwp is dropped out along with generation_mw
        apply_dropout(
            source_data["generation_input"],
            t0,
            dropout_delay=choose_delay(gen_config.input.dropout_minute_probabilities),
        )

    return


def validate_and_fill_source_nans(source_data: SourceDict[TArray], config: PVNetDataConfig) -> None:
    """Validate source values and fill NaNs according to configuration.

    Args:
        source_data: Input source arrays.
        config: PVNetDataConfig object.
    """
    if config.generation is not None:
        if config.generation.input is not None:
            validate_and_fill_nans(
                source_data["generation_input"],
                config.generation.input.fill_value,
                reject_nan=config.generation.input.reject_nan,
                source="generation_input",
            )
        if config.generation.target is not None:
            validate_and_fill_nans(
                source_data["generation_target"],
                fill_value=None,
                reject_nan=True,
                source="generation_target",
            )

    if config.satellite is not None:
        validate_and_fill_nans(
            source_data["sat"],
            config.satellite.fill_value,
            reject_nan=config.satellite.reject_nan,
            source="sat",
        )

    if config.nwp is not None:
        for nwp_name, nwp_config in config.nwp.items():
            validate_and_fill_nans(
                source_data["nwp"][nwp_name],
                nwp_config.fill_value,
                reject_nan=nwp_config.reject_nan,
                source=f"nwp/{nwp_name}",
            )


def preprocess_source_data(
    source_data: SourceDict[TArray],
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

    Updates the dictionary and its DataArray-like objects in-place, replacing entries or their
    `.data` as needed. The original NumPy arrays remain unchanged.

    Note: `source_data` is expected to already be loaded - see `materialise_data`.

    Args:
        source_data: Source arrays to preprocess
        t0: The init-time
        config: PVNetDataConfig object
        normalisation_arrays: Precomputed arrays from `build_normalisation_arrays`
    """
    diff_accumulated_nwp_channels(source_data, config)
    normalise_source_data(source_data, normalisation_arrays)
    apply_source_dropout(source_data, t0, config)
    validate_and_fill_source_nans(source_data, config=config)
