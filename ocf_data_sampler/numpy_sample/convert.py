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
        sample: Dictionary of xarray DataArrays, with the same structure as used inside
            the PVNet Dataset classes. Expected keys are any of:
            - "generation": DataArray of generation data
            - "satellite": DataArray of satellite data
            - "nwp": dict of DataArrays keyed by provider name (e.g. {"ukv": da, "ecmwf": da})
        t0: The t0 timestamp for this sample, used to compute datetime encodings and t0_idx

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
        t0_idx = _get_t0_idx(da.time_utc.values, t0)
        numpy_sample.update({
            "satellite_actual": da.values,
            "satellite_time_utc": da.time_utc.values.astype(float),
            "satellite_x_geostationary": da.x_geostationary.values,
            "satellite_y_geostationary": da.y_geostationary.values,
            "satellite_t0_idx": t0_idx,
        })

    if "nwp" in sample:
        numpy_sample["nwp"] = {}
        for provider, da in sample["nwp"].items():
            target_time_utc = da.init_time_utc.values + da.step.values
            t0_idx = _get_t0_idx(target_time_utc, t0)
            numpy_sample["nwp"][provider] = {
                "nwp": da.values,
                "nwp_channel_names": da.channel.values,
                "nwp_init_time_utc": da.init_time_utc.values.astype(float),
                "nwp_step": (da.step.values / np.timedelta64(1, "h")).astype(int),
                "nwp_target_time_utc": target_time_utc.astype(float),
                "nwp_t0_idx": t0_idx,
            }

    return numpy_sample


def _get_t0_idx(time_values: np.ndarray, t0: pd.Timestamp | np.datetime64) -> int:
    """Find the index of t0 in an array of time values."""
    t0_ns = pd.Timestamp(t0).value  # normalise to int nanoseconds regardless of input type
    return int((time_values.astype("datetime64[ns]").view("int64") == t0_ns).argmax())
