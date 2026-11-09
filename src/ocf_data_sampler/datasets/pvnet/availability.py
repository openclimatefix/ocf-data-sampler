"""Calculate source availability and build valid PVNet sample indices."""

from itertools import pairwise
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.config.model import (
    Generation,
    GenerationInputWindow,
    PVNetDataConfig,
    Satellite,
)
from ocf_data_sampler.datasets.pvnet.sample_index import ConcurrentSampleIndex, SampleIndex
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.select.time_periods import (
    fill_time_periods,
    find_contiguous_t0_periods,
    find_contiguous_t0_periods_nwp,
    intersect_time_periods,
)


def validate_requested_periods(requested_periods: list[tuple[str | None, str | None]]) -> None:
    """Validate non-overlapping sampling periods, allowing unbounded endpoints."""
    if len(requested_periods) == 0:
        raise ValueError("At least one time period must be provided")

    parsed_periods: list[tuple[np.datetime64 | None, np.datetime64 | None]] = []
    for start, end in requested_periods:
        start_dt = np.datetime64(start, "ns") if start is not None else None
        end_dt = np.datetime64(end, "ns") if end is not None else None
        if (
            (start_dt is not None and np.isnat(start_dt))
            or (end_dt is not None and np.isnat(end_dt))
            or (start_dt is not None and end_dt is not None and start_dt >= end_dt)
        ):
            raise ValueError(
                f"Invalid requested period [{start}, {end}): expected valid dates and start < end"
            )
        parsed_periods.append((start_dt, end_dt))

    sorted_periods = sorted(parsed_periods, key=lambda period: (period[0] is not None, period[0]))
    for (_, previous_end), (next_start, _) in pairwise(sorted_periods):
        if previous_end is None or next_start is None or next_start < previous_end:
            raise ValueError("requested_periods must not overlap")


def _required_interval_end_minutes(
    window: Satellite | GenerationInputWindow,
    dropout_mode: Literal["random", "minimal"],
) -> int:
    """End of a window, relative to t0, up to which data must be available.

    In minimal mode, data after the largest dropout cutoff can always be dropped out, so is not
    required. The first time in the window is always required.
    """
    if dropout_mode == "random" or window.dropout_minute_probabilities is None:
        return window.interval_end_minutes
    # Config validation already places every cutoff before the interval end
    return max(window.interval_start_minutes, -max(window.dropout_minute_probabilities))


def _find_non_generation_t0_periods(
    source_data: SourceDict[xr.DataArray],
    config: PVNetDataConfig,
    dropout_mode: Literal["random", "minimal"],
) -> pd.DataFrame:
    """Find overlapping valid t0 periods for NWP and satellite sources.

    Args:
        source_data: Input source arrays.
        config: PVNet data configuration
        dropout_mode: How dropout and NWP delays are chosen when sampling.

    Returns:
        A DataFrame containing the valid t0 time periods.
    """
    source_periods: list[pd.DataFrame] = []
    if config.nwp is not None:
        for nwp_name, nwp_config in config.nwp.items():
            da = source_data["nwp"][nwp_name]

            first_forecast_step = da["step"].values[0]
            last_forecast_step = da["step"].values[-1]

            # In minimal mode, NWP uses the most recent available init-time
            if nwp_config.delay_minute_probabilities is None or dropout_mode == "minimal":
                max_delay = minutes(0)
            else:
                max_delay = minutes(max(nwp_config.delay_minute_probabilities))

            # The last step of the forecast is lost if we have to diff channels
            if any(channel.accumulated for channel in nwp_config.channels):
                end_buffer = minutes(nwp_config.time_resolution_minutes)
            else:
                end_buffer = minutes(0)

            # Default to use max possible staleness unless specified in config
            if nwp_config.staleness_limit_minutes is None:
                staleness_limit = None
            else:
                staleness_limit = minutes(nwp_config.staleness_limit_minutes)

            nwp_periods = find_contiguous_t0_periods_nwp(
                init_times=da["init_time_utc"].values,
                interval_start=minutes(nwp_config.interval_start_minutes),
                interval_end=minutes(nwp_config.interval_end_minutes) + end_buffer,
                first_forecast_step=first_forecast_step,
                last_forecast_step=last_forecast_step,
                time_resolution=minutes(nwp_config.time_resolution_minutes),
                max_delay=max_delay,
                staleness_limit=staleness_limit,
            )

            if len(nwp_periods) == 0:
                raise ValueError(
                    f"No valid t0 periods found for NWP source {nwp_name} with the requested "
                    "configuration."
                )

            source_periods.append(nwp_periods)

    if config.satellite is not None:
        sat_periods = find_contiguous_t0_periods(
            source_data["sat"]["time_utc"].values,
            time_resolution=minutes(config.satellite.time_resolution_minutes),
            interval_start=minutes(config.satellite.interval_start_minutes),
            interval_end=minutes(_required_interval_end_minutes(config.satellite, dropout_mode)),
        )

        if len(sat_periods) == 0:
            raise ValueError(
                "No valid t0 periods found for satellite source with the requested configuration."
            )
        source_periods.append(sat_periods)

    intersected_periods = intersect_time_periods(source_periods)

    if len(intersected_periods) == 0:
        raise ValueError(f"No intersecting time periods found, {source_periods=}")

    return intersected_periods


def _find_generation_t0_periods(
    datetimes: NDArray[np.datetime64],
    generation_config: Generation,
    dropout_mode: Literal["random", "minimal"],
) -> pd.DataFrame:
    """Intersect valid t0 periods for the configured generation input and target windows.

    Args:
        datetimes: One-dimensional array of available generation timestamps, sorted and unique.
        generation_config: Generation configuration specifying the time resolution and windows.
        dropout_mode: How dropout is chosen when sampling.

    Returns:
        A DataFrame of periods valid for every configured window, with inclusive start_dt and
        exclusive end_dt bounds. Empty if no periods are valid.
    """
    window_periods: list[pd.DataFrame] = []
    for window in (generation_config.input, generation_config.target):
        if window is None:
            continue

        # Only the input window has dropout, since the target is what we predict
        if isinstance(window, GenerationInputWindow):
            interval_end_minutes = _required_interval_end_minutes(window, dropout_mode)
        else:
            interval_end_minutes = window.interval_end_minutes

        window_periods.append(
            find_contiguous_t0_periods(
                datetimes,
                time_resolution=minutes(generation_config.time_resolution_minutes),
                interval_start=minutes(window.interval_start_minutes),
                interval_end=minutes(interval_end_minutes),
            )
        )

    intersected_periods = intersect_time_periods(window_periods)

    return intersected_periods


def _filter_times_to_requested_periods(
    times: NDArray[np.datetime64],
    requested_periods: list[tuple[str | None, str | None]],
) -> NDArray[np.datetime64]:
    """Filter times to the requested periods.

    Args:
        times: One-dimensional array of timestamps to filter.
        requested_periods: Validated periods with inclusive starts and exclusive ends.
            A None boundary is unbounded in that direction.

    Returns:
        Timestamps within any requested period, preserving their order and dtype.
    """
    mask = np.zeros(len(times), dtype=bool)
    for start, end in requested_periods:
        period_mask = np.ones(len(times), dtype=bool)
        if start is not None:
            period_mask &= times >= np.datetime64(start)
        if end is not None:
            period_mask &= times < np.datetime64(end)
        mask |= period_mask
    return times[mask]


def _build_t0_times(
    available_periods: pd.DataFrame,
    t0_frequency: np.timedelta64,
    requested_periods: list[tuple[str | None, str | None]] | None,
) -> NDArray[np.datetime64]:
    """Fill valid periods on the sampling grid and apply optional requested periods."""
    t0_times = fill_time_periods(available_periods, freq=t0_frequency)
    if requested_periods is not None:
        t0_times = _filter_times_to_requested_periods(t0_times, requested_periods)
    return t0_times


def _build_t0_times_from_requested_periods(
    requested_periods: list[tuple[str | None, str | None]] | None,
    t0_frequency: np.timedelta64,
) -> NDArray[np.datetime64]:
    """Generate t0 times from bounded requested periods when no sources exist."""
    if not requested_periods or any(
        start is None or end is None for start, end in requested_periods
    ):
        raise ValueError(
            "When no data sources are configured, requested_periods must specify "
            "a finite start and end for every period."
        )

    requested_periods_df = pd.DataFrame(
        requested_periods, columns=["start_dt", "end_dt"], dtype="datetime64[ns]"
    )

    return fill_time_periods(requested_periods_df, freq=t0_frequency)


def build_sample_index(
    source_data: SourceDict[xr.DataArray],
    location_ids: list[str],
    config: PVNetDataConfig,
    requested_periods: list[tuple[str | None, str | None]] | None,
    dropout_mode: Literal["random", "minimal"],
) -> SampleIndex:
    """Construct a SampleIndex containing all valid t0 and location_id pairs.

    Args:
        source_data: Validated source arrays matching the configured sources.
        location_ids: Non-empty list of validated sampling location IDs, present in generation
            data when generation is configured.
        config: PVNet data configuration
        requested_periods: Validated periods with inclusive starts and exclusive ends.
            Bounds may be None only when data sources are available.
        dropout_mode: How dropout and NWP delays are chosen when sampling. In minimal mode,
            t0s only need data up to the largest dropout cutoff.
    """
    t0_frequency = minutes(config.sampling_grid.t0_resolution_minutes)

    # Without sources, requested periods define the sampling range.
    if config.satellite is None and config.nwp is None and config.generation is None:
        t0_times = _build_t0_times_from_requested_periods(requested_periods, t0_frequency)
        if len(t0_times) == 0:
            raise ValueError("No t0 times found for locations")

        return SampleIndex(
            t0=np.tile(t0_times, len(location_ids)),
            location_id=np.repeat(location_ids, len(t0_times)),
        )

    # Without generation, all locations share the same t0 times.
    if config.generation is None:
        periods = _find_non_generation_t0_periods(source_data, config, dropout_mode)
        t0_times = _build_t0_times(periods, t0_frequency, requested_periods)
        if len(t0_times) == 0:
            raise ValueError("No t0 times found for locations")
        return SampleIndex(
            t0=np.tile(t0_times, len(location_ids)),
            location_id=np.repeat(location_ids, len(t0_times)),
        )

    # Shared source periods apply to every location, so calculate them once.
    non_generation_periods = (
        _find_non_generation_t0_periods(source_data, config, dropout_mode)
        if config.satellite is not None or config.nwp is not None
        else None
    )

    # Generation availability can differ by location.
    t0_arrays: list[NDArray[np.datetime64]] = []
    location_id_arrays: list[NDArray[np.str_]] = []
    unavailable_location_ids: list[str] = []
    for location_id in location_ids:
        da_gen_loc = source_data["generation"].sel(location_id=location_id).dropna(dim="time_utc")

        generation_periods = _find_generation_t0_periods(
            da_gen_loc["time_utc"].values, config.generation, dropout_mode
        )
        if non_generation_periods is not None:
            available_periods = intersect_time_periods([non_generation_periods, generation_periods])
        else:
            available_periods = generation_periods

        t0_times = _build_t0_times(available_periods, t0_frequency, requested_periods)
        if len(t0_times) == 0:
            unavailable_location_ids.append(location_id)
        else:
            t0_arrays.append(t0_times.astype("datetime64[ns]"))
            location_id_arrays.append(np.full(len(t0_times), location_id))

    if len(unavailable_location_ids) > 0:
        raise ValueError(f"No valid t0 times found for locations: {unavailable_location_ids}")

    return SampleIndex(
        t0=np.concatenate(t0_arrays),
        location_id=np.concatenate(location_id_arrays),
    )


def build_concurrent_sample_index(
    source_data: SourceDict[xr.DataArray],
    config: PVNetDataConfig,
    requested_periods: list[tuple[str | None, str | None]] | None,
    dropout_mode: Literal["random", "minimal"],
) -> ConcurrentSampleIndex:
    """Construct an index of t0 times available at every selected location.

    Args:
        source_data: Validated source arrays matching the configured sources, with generation
            restricted to a non-empty set of validated requested locations.
        config: PVNet data configuration
        requested_periods: Validated periods with inclusive starts and exclusive ends.
            Bounds may be None only when data sources are available.
        dropout_mode: How dropout and NWP delays are chosen when sampling. In minimal mode,
            t0s only need data up to the largest dropout cutoff.
    """
    t0_frequency = minutes(config.sampling_grid.t0_resolution_minutes)

    # Without sources, requested periods define the sampling range.
    if config.generation is None and config.satellite is None and config.nwp is None:
        t0_times = _build_t0_times_from_requested_periods(requested_periods, t0_frequency)
        if len(t0_times) == 0:
            raise ValueError("No t0 times are available")
        return ConcurrentSampleIndex(t0=t0_times)

    # Without generation, all locations share the same t0 times.
    if config.generation is None:
        non_generation_periods = _find_non_generation_t0_periods(source_data, config, dropout_mode)
        t0_times = _build_t0_times(non_generation_periods, t0_frequency, requested_periods)
        if len(t0_times) == 0:
            raise ValueError("No t0 times found")
        return ConcurrentSampleIndex(t0=t0_times)

    # Shared source periods apply to every location, so calculate them once.
    non_generation_periods = (
        _find_non_generation_t0_periods(source_data, config, dropout_mode)
        if config.satellite is not None or config.nwp is not None
        else None
    )

    # All locations must have non-NaN generation data for each t0
    da_gen = source_data["generation"].dropna(dim="time_utc", how="any")

    generation_periods = _find_generation_t0_periods(
        da_gen["time_utc"].values, config.generation, dropout_mode
    )
    if non_generation_periods is not None:
        available_periods = intersect_time_periods([non_generation_periods, generation_periods])
    else:
        available_periods = generation_periods

    t0_times = _build_t0_times(available_periods, t0_frequency, requested_periods)

    if len(t0_times) == 0:
        raise ValueError("No t0 times are available at every requested location")

    return ConcurrentSampleIndex(t0=t0_times)
