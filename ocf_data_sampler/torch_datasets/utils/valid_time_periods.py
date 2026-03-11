"""Functions pertaining to finding valid time periods for the input data."""

import numpy as np
import pandas as pd

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.find_contiguous_time_periods import (
    find_contiguous_t0_periods,
    find_contiguous_t0_periods_nwp,
    intersection_of_multiple_dataframes_of_periods,
)
from ocf_data_sampler.time_utils import minutes


def find_valid_time_periods(datasets_dict: dict, config: Configuration) -> pd.DataFrame:
    """Find the t0 times where all of the requested input data is available.

    Args:
        datasets_dict: A dictionary of input datasets
        config: Configuration file
    """
    if not set(datasets_dict.keys()).issubset({"nwp", "sat", "generation"}):
        raise ValueError(f"Invalid keys in datasets_dict: {datasets_dict.keys()}")

    # Used to store contiguous time periods from each data source
    contiguous_time_periods: dict[str : pd.DataFrame] = {}
    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.input_data.nwp.items():
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

            contiguous_time_periods[f"nwp_{nwp_key}"] = time_periods

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        time_periods = find_contiguous_t0_periods(
            datasets_dict["sat"]["time_utc"].values,
            time_resolution=minutes(sat_config.time_resolution_minutes),
            interval_start=minutes(sat_config.interval_start_minutes),
            interval_end=minutes(sat_config.interval_end_minutes),
        )

        contiguous_time_periods["sat"] = time_periods

    if "generation" in datasets_dict:
        generation_config = config.input_data.generation

        time_periods = find_contiguous_t0_periods(
            datasets_dict["generation"]["time_utc"].values,
            time_resolution=minutes(generation_config.time_resolution_minutes),
            interval_start=minutes(generation_config.interval_start_minutes),
            interval_end=minutes(generation_config.interval_end_minutes),
        )

        contiguous_time_periods["generation"] = time_periods

    # just get the values (not the keys)
    contiguous_time_periods_values = list(contiguous_time_periods.values())

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_periods_values) > 1:
        valid_time_periods = intersection_of_multiple_dataframes_of_periods(
            contiguous_time_periods_values,
        )
    else:
        valid_time_periods = contiguous_time_periods_values[0]

    # check there are some valid time periods
    if len(valid_time_periods) == 0:
        raise ValueError(f"No valid time periods found, {contiguous_time_periods=}")

    return valid_time_periods
