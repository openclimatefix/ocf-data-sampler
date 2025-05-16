"""Slice datasets by time."""

import pandas as pd
import xarray as xr

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.dropout import apply_sampled_dropout_time
from ocf_data_sampler.select.select_time_slice import select_time_slice, select_time_slice_nwp
from ocf_data_sampler.utils import minutes


def slice_datasets_by_time(
    datasets_dict: dict,
    t0: pd.Timestamp,
    config: Configuration,
) -> dict:
    """Slice the dictionary of input data sources around a given t0 time.

    Args:
        datasets_dict: Dictionary of the input data sources
        t0: The init-time
        config: Configuration object.
    """
    sliced_datasets_dict = {}

    if "nwp" in datasets_dict:
        sliced_datasets_dict["nwp"] = {}

        for nwp_key, da_nwp in datasets_dict["nwp"].items():
            nwp_config = config.input_data.nwp[nwp_key]

            sliced_datasets_dict["nwp"][nwp_key] = select_time_slice_nwp(
                da_nwp,
                t0,
                time_resolution=minutes(nwp_config.time_resolution_minutes),
                interval_start=minutes(nwp_config.interval_start_minutes),
                interval_end=minutes(nwp_config.interval_end_minutes),
                dropout_timedeltas=minutes(nwp_config.dropout_timedeltas_minutes),
                dropout_frac=nwp_config.dropout_fraction,
                accum_channels=nwp_config.accum_channels,
            )

    if "sat" in datasets_dict:
        sat_config = config.input_data.satellite

        sliced_datasets_dict["sat"] = select_time_slice(
            datasets_dict["sat"],
            t0,
            time_resolution=minutes(sat_config.time_resolution_minutes),
            interval_start=minutes(sat_config.interval_start_minutes),
            interval_end=minutes(sat_config.interval_end_minutes),
        )

        # Apply the randomly sampled dropout
        sliced_datasets_dict["sat"] = apply_sampled_dropout_time(
            t0,
            dropout_timedeltas=minutes(sat_config.dropout_timedeltas_minutes),
            dropout_frac=sat_config.dropout_fraction,
            da=sliced_datasets_dict["sat"],
        )

    if "gsp" in datasets_dict:
        gsp_config = config.input_data.gsp

        da_gsp_past = select_time_slice(
            datasets_dict["gsp"],
            t0,
            time_resolution=minutes(gsp_config.time_resolution_minutes),
            interval_start=minutes(gsp_config.interval_start_minutes),
            interval_end=minutes(0),
        )

        # Dropout on the past GSP, but not the future GSP
        da_gsp_past = apply_sampled_dropout_time(
            t0,
            dropout_timedeltas=minutes(gsp_config.dropout_timedeltas_minutes),
            dropout_frac=gsp_config.dropout_fraction,
            da=da_gsp_past,
        )

        da_gsp_future = select_time_slice(
            datasets_dict["gsp"],
            t0,
            time_resolution=minutes(gsp_config.time_resolution_minutes),
            interval_start=minutes(gsp_config.time_resolution_minutes),
            interval_end=minutes(gsp_config.interval_end_minutes),
        )

        sliced_datasets_dict["gsp"] = xr.concat([da_gsp_past, da_gsp_future], dim="time_utc")

    if "site" in datasets_dict:
        site_config = config.input_data.site

        da_site_past = select_time_slice(
            datasets_dict["site"],
            t0,
            time_resolution=minutes(site_config.time_resolution_minutes),
            interval_start=minutes(site_config.interval_start_minutes),
            interval_end=minutes(0),
        )

        # Apply the randomly sampled dropout on the past site not the future
        da_site_past = apply_sampled_dropout_time(
            t0,
            dropout_timedeltas=minutes(site_config.dropout_timedeltas_minutes),
            dropout_frac=site_config.dropout_fraction,
            da=da_site_past,
        )

        da_site_future = select_time_slice(
            datasets_dict["site"],
            t0,
            time_resolution=minutes(site_config.time_resolution_minutes),
            interval_start=minutes(site_config.time_resolution_minutes),
            interval_end=minutes(site_config.interval_end_minutes),
        )

        sliced_datasets_dict["site"] = xr.concat([da_site_past, da_site_future], dim="time_utc")

    return sliced_datasets_dict
