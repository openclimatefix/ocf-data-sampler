"""Open multiple zarrs with TensorStore.

This extendds the functionality of xarray_tensorstore to open multiple zarr stores
"""

import os

import tensorstore as ts
import xarray as xr
from xarray_tensorstore import (
    _raise_if_mask_and_scale_used_for_data_vars,
    _TensorStoreAdapter,
    _zarr_spec_from_path,
)


def tensorstore_open_multi_zarrs(
    paths: list[str],
    data_vars: list[str],
    concat_axes: list[int],
    context: ts.Context,
    write: bool,
) -> dict[str, ts.TensorStore]:
    """Open multiple zarrs with TensorStore.

    Args:
        paths: List of paths to zarr stores.
        data_vars: List of data variable names to open.
        concat_axes: List of axes along which to concatenate the data variables.
        context: TensorStore context.
        write: Whether to open the stores for writing.
    """
    arrays_list = []
    for path in paths:
        specs = {k: _zarr_spec_from_path(os.path.join(path, k)) for k in data_vars}
        array_futures = {
          k: ts.open(spec, read=True, write=write, context=context)
          for k, spec in specs.items()
        }
        arrays_list.append({k: v.result() for k, v in array_futures.items()})

    arrays = {}
    for k, axis in zip(data_vars, concat_axes, strict=False):
        datasets = [d[k] for d in arrays_list]
        arrays[k] = ts.concat(datasets, axis=axis)

    return arrays


def open_zarrs(
    paths: list[str],
    concat_dim: str,
    *,
    context: ts.Context | None = None,
    mask_and_scale: bool = True,
    write: bool = False,
) -> xr.Dataset:
    """Open multiple zarrs with TensorStore.

    Args:
        paths: List of paths to zarr stores.
        concat_dim: Dimension along which to concatenate the data variables.
        context: TensorStore context.
        mask_and_scale: Whether to mask and scale the data.
        write: Whether to open the stores for writing.
    """
    if context is None:
        context = ts.Context()

    ds = xr.open_mfdataset(
        paths,
        concat_dim=concat_dim,
        combine="nested",
        mask_and_scale=mask_and_scale,
        decode_timedelta=True,
    )

    if mask_and_scale:
        # Data variables get replaced below with _TensorStoreAdapter arrays, which
        # don't get masked or scaled. Raising an error avoids surprising users with
        # incorrect data values.
        _raise_if_mask_and_scale_used_for_data_vars(ds)

    data_vars = list(ds.data_vars)

    concat_axes = [ds[v].dims.index(concat_dim) for v in data_vars]

    arrays = tensorstore_open_multi_zarrs(paths, data_vars, concat_axes, context, write)

    new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

    return ds.copy(data=new_data)
