"""Read, load, and convert PVNet source payloads to float32."""

from collections.abc import Iterator

import numpy as np
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.common.types import DataArrayLike
from ocf_data_sampler.common.xr_tensorstore import read as read_tensorstore
from ocf_data_sampler.datasets.pvnet.types import SourceDict


def _is_numpy_backed(da: xr.DataArray | LightDataArray) -> bool:
    """Check for NumPy-backed data without triggering a lazy xarray read."""
    data = da.data if isinstance(da, LightDataArray) else da.variable._data
    return isinstance(data, np.ndarray)


def _iter_source_arrays[TArray: DataArrayLike](
    source_data: SourceDict[TArray],
) -> Iterator[TArray]:
    """Iterate over source arrays, including named NWP sources."""
    for key in ("sat", "generation", "generation_input", "generation_target"):
        if key in source_data:
            yield source_data[key]
    if "nwp" in source_data:
        yield from source_data["nwp"].values()


def initiate_reads[TArray: xr.DataArray | LightDataArray](source_data: SourceDict[TArray]) -> None:
    """Initiate source reads in-place before waiting for any one source."""
    for v in _iter_source_arrays(source_data):
        if not _is_numpy_backed(v):
            if isinstance(v, LightDataArray):
                v.read()
            else:
                read_tensorstore(v)


def block_until_loaded[TArray: xr.DataArray | LightDataArray](
    source_data: SourceDict[TArray],
) -> None:
    """Block until source arrays are loaded into memory in-place."""
    for v in _iter_source_arrays(source_data):
        if not _is_numpy_backed(v):
            # .load() is in-place in both xarray.DataArray and LightDataArray
            v.load()


def convert_to_float32[TArray: DataArrayLike](
    source_data: SourceDict[TArray],
) -> None:
    """Assign float32 data to source DataArrays, preserving coordinates."""
    for value in _iter_source_arrays(source_data):
        value.data = value.data.astype(np.float32, copy=False)


def materialise_data[TArray: xr.DataArray | LightDataArray](
    source_data: SourceDict[TArray],
) -> None:
    """Read and load source arrays into memory, then convert their payloads to float32 in-place."""
    initiate_reads(source_data)
    block_until_loaded(source_data)
    convert_to_float32(source_data)
