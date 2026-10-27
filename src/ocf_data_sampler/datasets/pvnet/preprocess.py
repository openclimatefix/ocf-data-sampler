"""Functions for normalising, differencing, and filling missing values in PVNet input data."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import get_string_index
from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.config.model import Channel, PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import (
    NormalisationArrays,
    SourceDict,
)
from ocf_data_sampler.features.diff_channels import diff_channels_along_step
from ocf_data_sampler.features.normalisation import clip_and_standardise
from ocf_data_sampler.select.dropout import apply_dropout, choose_delay


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
            da.data = clip_and_standardise(
                da.data,
                clip_min=params.clip_min,
                clip_max=params.clip_max,
                mean=params.mean,
                std=params.std,
            )

    if "sat" in source_data:
        da = source_data["sat"]
        params = normalisation_arrays["sat"]
        da.data = clip_and_standardise(
            da.data,
            clip_min=params.clip_min,
            clip_max=params.clip_max,
            mean=params.mean,
            std=params.std,
        )

    for key in ("generation_input", "generation_target"):
        if key in source_data:
            normalise_generation_by_capacity(source_data[key])


def normalise_generation_by_capacity(da: TArray) -> None:
    """Rescale `generation_mw` to a capacity factor, leaving `capacity_mwp` unchanged.

    Replaces `.data` on the input DataArray without modifying shared source values.

    Zero capacity means no plant, so the capacity factor is taken as 0 rather than the undefined
    0/0. Emitting NaN instead would route it through the dropout fill, which signals missing data
    rather than a known-zero output.

    Args:
        da: Generation DataArray-like with a `gen_param` dimension
    """
    gen_params = da["gen_param"].values
    gen_idx = get_string_index(gen_params, "generation_mw")
    cap_idx = get_string_index(gen_params, "capacity_mwp")

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


def validate_and_fill_nans(
    da: TArray,
    fill_value: float | None,
    *,
    reject_nan: bool = False,
    reject_inf: bool = True,
    source: str,
) -> None:
    """Reject disallowed input values and optionally fill NaNs.

    Replaces `.data` only when filling NaNs, without modifying the original NumPy array.

    Args:
        da: Materialised DataArray-like to fill and validate.
        fill_value: Finite replacement for NaNs, or None to preserve them if permitted.
            Must be None when reject_nan is True.
        reject_nan: Whether NaNs in the input raise an error.
        reject_inf: Whether positive and negative infinity in the input raise an error.
        source: Source identifier included in validation errors.

    Raises:
        ValueError: A fill value is configured with reject_nan, or the input contains
            disallowed NaNs or infinities.
    """
    if reject_nan and fill_value is not None:
        raise ValueError(f"{source}: fill_value must be None when reject_nan is True")

    if reject_inf and np.isinf(da.data).any():
        raise ValueError(f"{source} contains infinities")

    if (fill_value is not None) or reject_nan:
        nan_mask = np.isnan(da.data)
        contains_nans = nan_mask.any()

        if contains_nans:
            if reject_nan:
                raise ValueError(f"{source} contains NaNs")
            if fill_value is not None:
                da.data = np.where(nan_mask, fill_value, da.data)


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
