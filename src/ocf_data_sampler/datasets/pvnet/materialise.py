"""Read, load, and convert PVNet source payloads to float32."""

import numpy as np
import xarray as xr
from xarray_tensorstore import read as xtr_read

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.types import TArray
from ocf_data_sampler.datasets.pvnet.types import SourceDict


def _is_numpy_backed(da: xr.DataArray | LightDataArray) -> bool:
    """Check for NumPy-backed data without triggering a lazy xarray read."""
    data = da.data if isinstance(da, LightDataArray) else da.variable._data
    return isinstance(data, np.ndarray)


def initiate_reads(source_data: SourceDict[TArray]) -> None:
    """Initiate source reads in-place before waiting for any one source."""
    # Kick off the tensorstore async reading
    for k, v in source_data.items():
        if isinstance(v, dict):
            initiate_reads(v)
        elif not _is_numpy_backed(v):
            if isinstance(v, LightDataArray):
                source_data[k].read()
            else:
                source_data[k] = xtr_read(v)


def block_until_loaded(source_data: SourceDict[TArray]) -> None:
    """Block until source arrays are loaded into memory in-place."""
    for v in source_data.values():
        if isinstance(v, dict):
            block_until_loaded(v)
        elif not _is_numpy_backed(v):
            # .load() is in-place in both xarray.DataArray and LightDataArray
            v.load()


def convert_to_float32(source_data: SourceDict[TArray]) -> None:
    """Assign float32 data to source DataArrays, preserving coordinates."""
    for value in source_data.values():
        if isinstance(value, dict):
            convert_to_float32(value)
        else:
            value.data = value.data.astype(np.float32, copy=False)


def materialise_data(source_data: SourceDict[TArray]) -> None:
    """Read and load source arrays into memory, then convert their payloads to float32 in-place."""
    initiate_reads(source_data)
    block_until_loaded(source_data)
    convert_to_float32(source_data)
