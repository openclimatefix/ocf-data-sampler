"""Convert a dictionary of xarray objects to a NumpySample."""

import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def convert_to_numpy_sample(
    sample: dict[str, xr.DataArray | dict[str, xr.DataArray]],
    t0: pd.Timestamp,
) -> NumpySample:
    """Convert a dictionary of xarray objects to a NumpySample.

    Args:
        sample: Dictionary of xarray DataArrays, with same structure as used inside
            PVNet Dataset classes. Expected keys are any of following:
            - "generation": DataArray of generation data
            - "sat": DataArray of satellite data
            - "nwp": dict of DataArrays by provider name (e.g. {"ukv": da, "ecmwf": da})
        t0: t0 timestamp for this sample, used to compute datetime encodings and t0_idx

    Returns:
        NumpySample dictionary with all modalities merged
    """
    numpy_sample: NumpySample = {}

    if "generation" in sample:
        da = sample["generation"]
        t0_idx = _get_t0_idx(da.time_utc.values, t0)
        numpy_sample.update({
            "generation": da.values,
            "capacity_mwp": da.capacity_mwp.values[0],
            "time_utc": da["time_utc"].values.astype(float),
            "t0_idx": t0_idx,
            "longitude": float(da.longitude.values),
            "latitude": float(da.latitude.values),
        })

    if "sat" in sample:
        da = sample["sat"]
        numpy_sample.update({
            "satellite_actual": da.values,
            "satellite_time_utc": da.time_utc.values.astype(float),
            "satellite_x_geostationary": da.x_geostationary.values,
            "satellite_y_geostationary": da.y_geostationary.values,
        })

    if "nwp" in sample:
        numpy_sample["nwp"] = {}
        for provider, da in sample["nwp"].items():
            target_time_utc = da.init_time_utc.values + da.step.values
            numpy_sample["nwp"][provider] = {
                "nwp": da.values,
                "nwp_channel_names": da.channel.values,
                "nwp_init_time_utc": da.init_time_utc.values.astype(float),
                "nwp_step": (da.step.values / np.timedelta64(1, "h")).astype(int),
                "nwp_target_time_utc": target_time_utc.astype(float),
            }

    return numpy_sample


def _get_t0_idx(time_values: np.ndarray, t0: pd.Timestamp | np.datetime64) -> int:
    """Find the index of t0 in an array of time values."""
    time_values = time_values.astype("datetime64[ns]")
    t0 = np.datetime64(t0, "ns")
    return int(np.searchsorted(time_values, t0, side="left"))
