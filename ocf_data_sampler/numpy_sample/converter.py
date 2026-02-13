"""Unified conversion function to create a NumpySample from PVNet-style xarray dict.

This is intended to replace:
- numpy_sample/generation.py
- numpy_sample/nwp.py
- numpy_sample/satellite.py

The output format MUST remain identical to the previous conversion functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

    from ocf_data_sampler.numpy_sample.common_types import NumpySample

def convert_xarray_dict_to_numpy_sample(
    xr_dict: dict[str, Any],
    *,
    t0_indices: dict[str, int] | None = None,
) -> NumpySample:
    """Convert PVNet-style dictionary of xarray objects into a single NumpySample.

    Args:
        xr_dict:
            Dictionary of xarray objects. Expected structure:
            {
              "generation": xr.DataArray,
              "satellite": xr.DataArray,
              "nwp": {
                  "ukv": xr.DataArray,
                  "gfs": xr.DataArray,
              },
              ...
            }

        t0_indices:
            Optional dict of t0 indices for each modality.

    Example:
              {
                "generation": 30,
                "satellite": 12,
                "ukv": 4,
                "gfs": 4,
                "nwp": 4,   # optional fallback applied to all providers
              }

    Returns:
        NumpySample dict.

    Notes:
        This function MUST NOT change the output keys/format compared to the
        old per-modality conversion functions.
    """
    sample: NumpySample = {}

    t0_indices = t0_indices or {}


    # Generation
    if "generation" in xr_dict and xr_dict["generation"] is not None:
        da: xr.DataArray = xr_dict["generation"]

        sample["generation"] = da.values
        sample["capacity_mwp"] = da.capacity_mwp.values[0]
        sample["time_utc"] = da["time_utc"].values.astype(float)

        if "generation" in t0_indices:
            sample["t0_idx"] = t0_indices["generation"]

    # Satellite
    if "satellite" in xr_dict and xr_dict["satellite"] is not None:
        da: xr.DataArray = xr_dict["satellite"]

        sample["satellite_actual"] = da.values
        sample["satellite_time_utc"] = da.time_utc.values.astype(float)
        sample["satellite_x_geostationary"] = da.x_geostationary.values
        sample["satellite_y_geostationary"] = da.y_geostationary.values

        if "satellite" in t0_indices:
            sample["satellite_t0_idx"] = t0_indices["satellite"]

    # NWP (nested by provider)
    if "nwp" in xr_dict and xr_dict["nwp"] is not None:
        sample["nwp"] = {}

        nwp_dict: dict[str, xr.DataArray] = xr_dict["nwp"]

        for provider_name, da in nwp_dict.items():
            provider_sample: dict[str, np.ndarray] = {}

            provider_sample["nwp"] = da.values
            provider_sample["nwp_channel_names"] = da.channel.values
            provider_sample["nwp_init_time_utc"] = da.init_time_utc.values.astype(float)
            provider_sample["nwp_step"] = (da.step.values / 3600).astype(int)
            provider_sample["nwp_target_time_utc"] = (
                da.init_time_utc.values + da.step.values
            ).astype(float)

            # Match old behavior: allow setting t0_idx for NWP providers.
            if provider_name in t0_indices:
                provider_sample["nwp_t0_idx"] = t0_indices[provider_name]
            elif "nwp" in t0_indices:
                provider_sample["nwp_t0_idx"] = t0_indices["nwp"]

            sample["nwp"][provider_name] = provider_sample

    return sample
