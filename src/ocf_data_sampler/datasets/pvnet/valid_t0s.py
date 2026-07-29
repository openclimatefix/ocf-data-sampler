"""Functions pertaining to finding valid time periods for the input data."""

import numpy as np
import pandas as pd

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.config.model import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.select.time_periods import (
    find_contiguous_t0_periods,
    find_contiguous_t0_periods_nwp,
    intersect_time_periods,
)


def find_valid_time_periods(
    datasets_dict: SourceDict,
    config: PVNetDataConfig,
) -> pd.DataFrame:
    """Find the t0 times where all of the requested input data is available.

    Args:
        datasets_dict: A dictionary of input datasets
        config: PVNetDataConfig file

    Returns:
        A DataFrame containing the valid t0 time periods.
    """
    contiguous_time_periods: list[pd.DataFrame] = []
    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.nwp.items():
            da = datasets_dict["nwp"][nwp_key]

            # Extract the max extents of the forecast steps
            first_forecast_step = da["step"].values[0]
            last_forecast_step = da["step"].values[-1]

            if nwp_config.dropout_timedeltas_minutes==[]:
                max_dropout = minutes(0)
            else:
                max_dropout = minutes(np.max(np.abs(nwp_config.dropout_timedeltas_minutes)))

            # The last step of the forecast is lost if we have to diff channels
            if len(nwp_config.accum_channels) > 0:
                end_buffer = minutes(nwp_config.time_resolution_minutes)
            else:
                end_buffer = minutes(0)

            # Default to use max possible staleness unless specified in config
            if nwp_config.max_staleness_minutes is None:
                max_staleness = None
            else:
                max_staleness = minutes(nwp_config.max_staleness_minutes)

            time_periods = find_contiguous_t0_periods_nwp(
                init_times=da["init_time_utc"].values,
                interval_start=minutes(nwp_config.interval_start_minutes),
                interval_end=minutes(nwp_config.interval_end_minutes)+end_buffer,
                first_forecast_step=first_forecast_step,
                last_forecast_step=last_forecast_step,
                max_dropout=max_dropout,
                max_staleness=max_staleness,
            )

            if len(time_periods) == 0:
                raise ValueError(f"No valid t0 periods found for {nwp_key} NWP data")

            contiguous_time_periods.append(time_periods)

    if "sat" in datasets_dict:
        time_periods = find_contiguous_t0_periods(
            datasets_dict["sat"]["time_utc"].values,
            time_resolution=minutes(config.satellite.time_resolution_minutes),
            interval_start=minutes(config.satellite.interval_start_minutes),
            interval_end=minutes(config.satellite.interval_end_minutes),
        )

        contiguous_time_periods.append(time_periods)

        if len(time_periods) == 0:
            raise ValueError("No valid t0 periods found for satellite data")

    if "generation" in datasets_dict:
        for window_config in (config.generation.input, config.generation.target):
            if window_config is None:
                continue

            time_periods = find_contiguous_t0_periods(
                datasets_dict["generation"]["time_utc"].values,
                time_resolution=minutes(config.generation.time_resolution_minutes),
                interval_start=minutes(window_config.interval_start_minutes),
                interval_end=minutes(window_config.interval_end_minutes),
            )

            if len(time_periods) == 0:
                raise ValueError("No valid t0 periods found for generation data")

            contiguous_time_periods.append(time_periods)

    # Find joint overlapping contiguous time periods
    valid_time_periods = intersect_time_periods(contiguous_time_periods)


    # check there are some valid time periods
    if len(valid_time_periods) == 0:
        raise ValueError(f"No valid time periods found, {contiguous_time_periods=}")

    return valid_time_periods
