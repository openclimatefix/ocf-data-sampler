import numpy as np
import pandas as pd

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.find_contiguous_time_periods import find_contiguous_t0_periods_nwp, \
    find_contiguous_t0_periods, intersection_of_multiple_dataframes_of_periods
from ocf_data_sampler.time_functions import minutes


def find_valid_time_periods(
    datasets_dict: dict,
    config: Configuration,
):
    """Find the t0 times where all of the requested input data is available

    Args:
        datasets_dict: A dictionary of input datasets
        config: Configuration file
    """

    assert set(datasets_dict.keys()).issubset({"nwp", "sat", "gsp"})

    contiguous_time_periods: dict[str: pd.DataFrame] = {}  # Used to store contiguous time periods from each data source

    if "nwp" in datasets_dict:
        for nwp_key, nwp_config in config.input_data.nwp.items():

            da = datasets_dict["nwp"][nwp_key]

            if nwp_config.dropout_timedeltas_minutes is None:
                max_dropout = minutes(0)
            else:
                max_dropout = minutes(np.max(np.abs(nwp_config.dropout_timedeltas_minutes)))

            if nwp_config.max_staleness_minutes is None:
                max_staleness = None
            else:
                max_staleness = minutes(nwp_config.max_staleness_minutes)

            # The last step of the forecast is lost if we have to diff channels
            if len(nwp_config.accum_channels) > 0:
                end_buffer = minutes(nwp_config.time_resolution_minutes)
            else:
                end_buffer = minutes(0)

            # This is the max staleness we can use considering the max step of the input data
            max_possible_staleness = (
                pd.Timedelta(da["step"].max().item())
                - minutes(nwp_config.forecast_minutes)
                - end_buffer
            )

            # Default to use max possible staleness unless specified in config
            if max_staleness is None:
                max_staleness = max_possible_staleness
            else:
                # Make sure the max acceptable staleness isn't longer than the max possible
                assert max_staleness <= max_possible_staleness

            time_periods = find_contiguous_t0_periods_nwp(
                datetimes=pd.DatetimeIndex(da["init_time_utc"]),
                history_duration=minutes(nwp_config.history_minutes),
                max_staleness=max_staleness,
                max_dropout=max_dropout,
            )

            contiguous_time_periods[f'nwp_{nwp_key}'] = time_periods

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(datasets_dict["sat"]["time_utc"]),
            sample_period_duration=minutes(sat_config.time_resolution_minutes),
            history_duration=minutes(sat_config.history_minutes),
            forecast_duration=minutes(sat_config.forecast_minutes),
        )

        contiguous_time_periods['sat'] = time_periods

    if "gsp" in datasets_dict:
        gsp_config = config.input_data.gsp

        time_periods = find_contiguous_t0_periods(
            pd.DatetimeIndex(datasets_dict["gsp"]["time_utc"]),
            sample_period_duration=minutes(gsp_config.time_resolution_minutes),
            history_duration=minutes(gsp_config.history_minutes),
            forecast_duration=minutes(gsp_config.forecast_minutes),
        )

        contiguous_time_periods['gsp'] = time_periods

    # just get the values (not the keys)
    contiguous_time_periods_values = list(contiguous_time_periods.values())

    # Find joint overlapping contiguous time periods
    if len(contiguous_time_periods_values) > 1:
        valid_time_periods = intersection_of_multiple_dataframes_of_periods(
            contiguous_time_periods_values
        )
    else:
        valid_time_periods = contiguous_time_periods_values[0]

    # check there are some valid time periods
    if len(valid_time_periods) == 0:
        raise ValueError(f"No valid time periods found, {contiguous_time_periods=}")

    return valid_time_periods
