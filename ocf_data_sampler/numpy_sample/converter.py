"""
Unified conversion of xarray DataArrays into a single NumpySample.

This replaces the modality-specific numpy conversion functions and removes the need
for scattered *SampleKey classes. The returned sample is structured by modality,
with shared fields stored under `metadata`.
"""

from typing import Callable
import xarray as xr
from ocf_data_sampler.numpy_sample.common_types import NumpySample


def _convert_generation(da: xr.DataArray, sample: NumpySample) -> None:
    """Convert generation DataArray into numpy sample."""
    sample["generation"] = {
        "values": da.values,
        "capacity_mwp": da.capacity_mwp.values[0],
    }

    # Use generation as the reference time axis
    sample["metadata"]["time_utc"] = da.time_utc.values.astype(float)


def _convert_nwp(da: xr.DataArray, sample: NumpySample) -> None:
    """Convert NWP DataArray into numpy sample."""
    sample["nwp"] = {
        "values": da.values,
        "channel_names": da.channel.values,
        "init_time_utc": da.init_time_utc.values.astype(float),
        "step_hours": (da.step.values / 3600).astype(int),
        "target_time_utc": (
            da.init_time_utc.values + da.step.values
        ).astype(float),
    }


def _convert_satellite(da: xr.DataArray, sample: NumpySample) -> None:
    """Convert satellite DataArray into numpy sample."""
    sample["satellite"] = {
        "values": da.values,
        "time_utc": da.time_utc.values.astype(float),
        "x_geostationary": da.x_geostationary.values,
        "y_geostationary": da.y_geostationary.values,
    }


# Registry of supported modalities
_CONVERTERS: dict[str, Callable[[xr.DataArray, NumpySample], None]] = {
    "generation": _convert_generation,
    "nwp": _convert_nwp,
    "satellite": _convert_satellite,
}


def convert_xarray_dict_to_numpy_sample(
    data: dict[str, xr.DataArray],
    *,
    t0_idx: int | None = None,
) -> NumpySample:
    """
    Convert a dictionary of xarray DataArrays into a unified NumpySample.

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
        "metadata": {
            "t0_idx": t0_idx,
        }
    }

    for modality, da in data.items():
        if modality not in _CONVERTERS:
            raise KeyError(f"Unsupported modality '{modality}'")

        _CONVERTERS[modality](da, sample)

    return sample
