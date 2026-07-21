"""Utilities for loading TensorStore data into Xarray.

This module uses and adapts internal functions from the Google xarray-tensorstore project [1],
licensed under the Apache License, Version 2.0. See [2] for details.

Modifications copyright 2025 Open Climate Fix. Licensed under the MIT License.

Modifications from the original include:
- Adding support for opening multiple zarr files as a single xarray object
- Support for zarr 3 -> https://github.com/google/xarray-tensorstore/pull/22

References:
    [1] https://github.com/google/xarray-tensorstore
    [2] https://www.apache.org/licenses/LICENSE-2.0
"""

import logging
import os
import re
from glob import glob, has_magic
from typing import Any, TypeAlias, cast

import tensorstore as ts
import xarray as xr
import zarr
from xarray_tensorstore import (
    _DEFAULT_STORAGE_DRIVER,
    _raise_if_mask_and_scale_used_for_data_vars,
    _TensorStoreAdapter,
)

logger = logging.getLogger(__name__)

ZarrPath: TypeAlias = str | os.PathLike[str]
ZarrSource: TypeAlias = ZarrPath | list[ZarrPath] | tuple[ZarrPath, ...]


def _zarr_spec_from_path(path: str, zarr_format: int) -> dict[str, Any]:
    if re.match(r"\w+\://", path):  # path is a URI
        kv_store: str | dict[str, str] = path
    else:
        kv_store = {"driver": _DEFAULT_STORAGE_DRIVER, "path": path}
    return {"driver": f"zarr{zarr_format}", "kvstore": kv_store}


def _get_data_variable_array_futures(
    path: str,
    context: ts.Context | None,
    variables: list[str],
) -> dict[str, ts.Future[ts.TensorStore]]:
    """Open all data variables in a zarr group and return futures.

    Args:
        path: path or URI to zarr group to open.
        context: TensorStore configuration options to use when opening arrays.
        variables: The variables in the zarr groupto open.
    """
    zarr_format = zarr.open(path).metadata.zarr_format
    specs = {k: _zarr_spec_from_path(os.path.join(path, k), zarr_format) for k in variables}
    return {k: ts.open(spec, read=True, write=False, context=context) for k, spec in specs.items()}


def _tensorstore_open_zarrs(
    paths: list[str],
    data_vars: list[str],
    concat_axes: list[int],
    context: ts.Context,
) -> dict[str, ts.TensorStore]:
    """Open multiple zarrs with TensorStore.

    Args:
        paths: List of paths to zarr stores.
        data_vars: List of data variable names to open.
        concat_axes: List of axes along which to concatenate the data variables.
        context: TensorStore context.
    """
    # Open all the variables from all the datasets - returned as futures
    array_futures_list: list[dict[str, ts.Future[ts.TensorStore]]] = []
    for path in paths:
        array_futures_list.append(_get_data_variable_array_futures(path, context, data_vars))

    # Wait for the async open operations
    arrays_list: list[dict[str, ts.TensorStore]] = [
        {k: future.result() for k, future in array_futures.items()}
        for array_futures in array_futures_list
    ]

    # Concatenate each of the variables along the required axis
    arrays: dict[str, ts.TensorStore] = {}
    for k, axis in zip(data_vars, concat_axes, strict=True):
        variable_arrays = [d[k] for d in arrays_list]
        arrays[k] = ts.concat(variable_arrays, axis=axis)

    return arrays


def open_zarr_paths(zarr_path: ZarrSource, concat_dim: str | None = None) -> xr.Dataset:
    """Open one or more Zarr stores using TensorStore.

    Args:
        zarr_path: A path, local glob pattern, or sequence of paths.
        concat_dim: Dimension along which multiple stores are concatenated.
    """
    if isinstance(zarr_path, str | os.PathLike):
        path = os.fspath(zarr_path)
        if not has_magic(path):
            return _open_single_zarr(path)
        paths = sorted(glob(path))
    else:
        paths = [os.fspath(path) for path in zarr_path]

    if not paths:
        raise ValueError(f"No Zarr stores found for {zarr_path!r}")

    if len(paths) == 1:
        return _open_single_zarr(paths[0])

    if concat_dim is None:
        raise ValueError("`concat_dim` must be specified when opening multiple Zarr stores")

    return _open_and_concat_zarrs(paths, concat_dim)


def _open_single_zarr(
    path: str,
    context: ts.Context | None = None,
    mask_and_scale: bool = True,
) -> xr.Dataset:
    """Open an xarray.Dataset from zarr using TensorStore.

    Args:
        path: path or URI to zarr group to open.
        context: TensorStore configuration options to use when opening arrays.
        mask_and_scale: if True (default), attempt to apply masking and scaling like
          xarray.open_zarr(). This is only supported for coordinate variables and
          otherwise will raise an error.

    Returns:
        Dataset with all data variables opened via TensorStore.
    """
    if context is None:
        context = ts.Context()

    # Avoid using dask by settung `chunks=None`
    ds = xr.open_zarr(path, chunks=None, mask_and_scale=mask_and_scale, consolidated=False)

    if mask_and_scale:
        _raise_if_mask_and_scale_used_for_data_vars(ds)

    # Open all data variables using tensorstore - returned as futures
    data_vars = list(ds.data_vars)
    array_futures = _get_data_variable_array_futures(path, context, data_vars)

    # Wait for the async open operations
    arrays = {k: future.result() for k, future in array_futures.items()}

    # Adapt the tensorstore arrays and plug them into the xarray object
    new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

    return cast("xr.Dataset", ds.copy(data=new_data))


def _open_and_concat_zarrs(
    paths: list[str],
    concat_dim: str,
    context: ts.Context | None = None,
    mask_and_scale: bool = True,
) -> xr.Dataset:
    """Open multiple zarrs with TensorStore.

    Args:
        paths: List of paths to zarr stores.
        concat_dim: Dimension along which to concatenate the data variables.
        context: TensorStore context.
        mask_and_scale: Whether to mask and scale the data.

    Returns:
        Concatenated Dataset with all data variables opened via TensorStore.
    """
    if context is None:
        context = ts.Context()

    ds_list = [
        xr.open_zarr(p, mask_and_scale=mask_and_scale, decode_timedelta=True, consolidated=False)
        for p in paths
    ]
    try:
        ds = xr.concat(
            ds_list,
            dim=concat_dim,
            data_vars="minimal",
            compat="equals",
            combine_attrs="drop_conflicts",
            join="exact",
        )
    except ValueError:
        logger.warning(
            f"Coordinate mismatch found when opening paths {paths}. Opening with `join='override'` "
            "to ignore coordinate mismatches. THIS MAY CAUSE UNEXPECTED BEHAVIOUR.",
        )
        ds = xr.concat(
            ds_list,
            dim=concat_dim,
            data_vars="minimal",
            compat="equals",
            combine_attrs="drop_conflicts",
            join="override",
        )

    if mask_and_scale:
        _raise_if_mask_and_scale_used_for_data_vars(ds)

    # Find the axis along which each data array must be concatenated
    data_vars = list(ds.data_vars)
    concat_axes = [ds[v].dims.index(concat_dim) for v in data_vars]

    # Open and concat all zarrs so each variables is a single TensorStore array
    arrays = _tensorstore_open_zarrs(paths, data_vars, concat_axes, context)

    # Plug the arrays into the xarray object
    new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

    return cast("xr.Dataset", ds.copy(data=new_data))
