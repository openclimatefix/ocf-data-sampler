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
import os.path
import re
from typing import Any

import tensorstore as ts
import xarray as xr
import zarr
from xarray_tensorstore import (
    _DEFAULT_STORAGE_DRIVER,
    _raise_if_mask_and_scale_used_for_data_vars,
    _TensorStoreAdapter,
)

logger = logging.getLogger(__name__)


def _get_data_var_names(ds: xr.Dataset) -> list[str]:
    data_vars: list[str] = []
    for name in ds.data_vars:
        if not isinstance(name, str):
            raise TypeError(f"Zarr data variable names must be strings, got {name!r}")
        data_vars.append(name)
    return data_vars


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


def open_zarr(
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
    ds: xr.Dataset = xr.open_zarr(
        path, chunks=None, mask_and_scale=mask_and_scale, consolidated=False
    )

    if mask_and_scale:
        _raise_if_mask_and_scale_used_for_data_vars(ds)

    # Open all data variables using tensorstore - returned as futures
    data_vars = _get_data_var_names(ds)
    array_futures = _get_data_variable_array_futures(path, context, data_vars)

    # Wait for the async open operations
    arrays = {k: future.result() for k, future in array_futures.items()}

    # Adapt the tensorstore arrays and plug them into the xarray object
    new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

    return ds.copy(data=new_data)


def open_zarrs(
    paths: list[str],
    concat_dim: str,
    context: ts.Context | None = None,
    mask_and_scale: bool = True,
    data_source: str = "unknown",
) -> xr.Dataset:
    """Open multiple zarrs with TensorStore.

    Args:
        paths: List of paths to zarr stores.
        concat_dim: Dimension along which to concatenate the data variables.
        context: TensorStore context.
        mask_and_scale: Whether to mask and scale the data.
        data_source: Which data source is being opened. Used for warning context.

    Returns:
        Concatenated Dataset with all data variables opened via TensorStore.
    """
    if context is None:
        context = ts.Context()

    ds_list: list[xr.Dataset] = [
        xr.open_zarr(p, mask_and_scale=mask_and_scale, decode_timedelta=True, consolidated=False)
        for p in paths
    ]
    ds: xr.Dataset
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
            f"Coordinate mismatch found in {data_source} input data. Opening with "
            "`join='override'` to ignore coordinate mismatches. THIS MAY CAUSE UNEXPECTED "
            "BEHAVIOUR!",
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
    data_vars = _get_data_var_names(ds)
    concat_axes = [ds[v].dims.index(concat_dim) for v in data_vars]

    # Open and concat all zarrs so each variables is a single TensorStore array
    arrays = _tensorstore_open_zarrs(paths, data_vars, concat_axes, context)

    # Plug the arrays into the xarray object
    new_data = {k: _TensorStoreAdapter(v) for k, v in arrays.items()}

    return ds.copy(data=new_data)
