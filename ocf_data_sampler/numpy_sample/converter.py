"""Unified conversion of xarray DataArrays into a single NumpySample.

This replaces the modality-specific numpy conversion functions and removes the need
for scattered *SampleKey classes. The returned sample is structured by modality,
with shared fields stored under `metadata`.
"""

from collections.abc import Callable

import numpy as np
import xarray as xr

from ocf_data_sampler.numpy_sample.common_types import NumpySample


def _convert_generation(da: xr.DataArray, sample: NumpySample) -> None:
    """Convert generation DataArray into numpy sample."""
    sample["generation"] = da.values
    sample["capacity_mwp"] = da.capacity_mwp.values[0]

    # Keep original datetime64 values for downstream functions that expect datetimes
    sample["time_utc"] = _datetime_or_timedelta_to_seconds(da.time_utc.values)


def _convert_nwp(
    nwp_dict_or_da: "dict[str, xr.DataArray] | xr.DataArray",
    sample: NumpySample,
) -> None:
    """Convert dict of NWP DataArrays (or single DataArray) into numpy sample."""
    if isinstance(nwp_dict_or_da, xr.DataArray):
        nwp_dict = {nwp_dict_or_da.name or "nwp": nwp_dict_or_da}
    else:
        nwp_dict = nwp_dict_or_da

    nwp_samples = {}

    for nwp_key, da in nwp_dict.items():
        # Provide legacy keys expected elsewhere in the codebase/tests.
        nwp_samples[nwp_key] = {
            "nwp": da.values,
            "nwp_channel_names": da.channel.values,
            "nwp_init_time_utc": _datetime_or_timedelta_to_seconds(
                da.init_time_utc.values,
            ),
            "nwp_step": (
                _datetime_or_timedelta_to_seconds(da.step.values) / 3600
            ).astype(int),
            "nwp_target_time_utc": _datetime_or_timedelta_to_seconds(
                da.init_time_utc.values + da.step.values,
            ),
        }

    sample["nwp"] = nwp_samples



def _convert_satellite(da: xr.DataArray, sample: NumpySample) -> None:
    """Convert satellite DataArray into numpy sample."""
    # Backwards-compatible top-level array key expected by callers/tests
    sample["satellite_actual"] = da.values

    # Newer structured representation kept under `satellite`
    sample["satellite"] = {
        "values": da.values,
        "time_utc": _datetime_or_timedelta_to_seconds(da.time_utc.values),
        "x_geostationary": da.x_geostationary.values,
        "y_geostationary": da.y_geostationary.values,
    }


def _datetime_or_timedelta_to_seconds(arr: np.ndarray) -> np.ndarray:
    """Convert numpy datetime64 or timedelta64 array to float seconds since epoch.

    Returns a float numpy array with seconds.
    """
    # Handle scalar datetime64/timedelta64
    if np.isscalar(arr) or arr.ndim == 0:
        arr = np.array([arr])
        was_scalar = True
    else:
        was_scalar = False

    # IMPORTANT: "M" is for datetime64, "m" is for timedelta64
    if arr.dtype.kind == "M":
        # datetime64 - convert to seconds since epoch
        result = arr.astype("datetime64[ns]").astype("int64") / 1e9
    elif arr.dtype.kind == "m":
        # timedelta64 - convert to seconds duration
        result = arr.astype("timedelta64[ns]").astype("int64") / 1e9
    else:
        # Fallback: try to convert to float directly
        result = arr.astype(float)

    return result[0] if was_scalar else result


# Registry of supported modalities
_CONVERTERS: dict[str, Callable[[xr.DataArray, NumpySample], None]] = {
    "generation": _convert_generation,
    "nwp": _convert_nwp,
    "satellite": _convert_satellite,
    # Backwards-compatible alias used throughout the codebase
    "sat": _convert_satellite,
}


def convert_xarray_dict_to_numpy_sample(
    data: dict[str, xr.DataArray],
    *,
    t0_idx: int | None = None,
) -> NumpySample:
    """Convert a dictionary of xarray DataArrays into a unified NumpySample.

    Args:
        data:
            Dictionary mapping modality name to xarray DataArray.
            Expected keys include: "generation", "nwp", "satellite".
        t0_idx:
            Optional index of the t0 timestamp, shared across modalities.

    Returns:
        NumpySample:
            A nested dictionary organised by modality with shared metadata.
    """
    sample: NumpySample = {
        "t0_idx": t0_idx,
    }

    for modality, da in data.items():
        if modality not in _CONVERTERS:
            raise KeyError(f"Unsupported modality '{modality}'")

        _CONVERTERS[modality](da, sample)

    return sample
