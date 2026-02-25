"""Convert a dictionary of xarray objects to a NumpySample."""

import numpy as np
import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def convert_to_numpy_sample(
    sample: dict[str, xr.DataArray | dict[str, xr.DataArray]],
    t0_idx: int,
) -> NumpySample:
    """Convert a dictionary of xarray objects to a NumpySample.

    Args:
        sample: Dictionary of xarray DataArrays, with same structure as used inside
            PVNet Dataset classes. Expected keys are any of following:
            - "generation": DataArray of generation data
            - "sat": DataArray of satellite data
            - "nwp": dict of DataArrays by provider name (e.g. {"ukv": da, "ecmwf": da})
        t0_idx: Index of t0 within generation

    Returns:
        NumpySample dictionary with all modalities merged
    """
    numpy_sample: NumpySample = {}

    if "generation" in sample:
        da = sample["generation"]

        generation_values = da.sel(gen_param="generation_mw").values
        capacity_value = da.sel(gen_param="capacity_mwp").values[0]

        if capacity_value!=0:
            generation_values = generation_values/capacity_value

        numpy_sample.update({
            "generation": generation_values,
            "capacity_mwp": capacity_value,
            "time_utc": da["time_utc"].values.astype(float),
            "t0_idx": int(t0_idx),
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
