"""Utility functions for working with xarray objects."""

import numpy as np
import xarray as xr
from glob import glob
from typing import Literal

import xarray as xr

from ocf_data_sampler.common.tensorstore import open_zarr, open_zarrs
from ocf_data_sampler.common.indexing import assert_values_unique_increasing


def make_spatial_coords_increasing(ds: xr.Dataset, x_coord: str, y_coord: str) -> xr.Dataset:
    """Make sure the spatial coordinates are in increasing order.

    Args:
        ds: Xarray Dataset
        x_coord: Name of the x coordinate
        y_coord: Name of the y coordinate
    """
    # Make sure the coords are in increasing order
    if ds[x_coord][0] > ds[x_coord][-1]:
        ds = ds.isel({x_coord: slice(None, None, -1)})
        # Below we the coord values so we don't have numpy array with negative strides
        # Numpy arrays with negative strides cannot be converted to torch Tensor
        ds[x_coord] = np.ascontiguousarray(ds[x_coord].values)
    if ds[y_coord][0] > ds[y_coord][-1]:
        ds = ds.isel({y_coord: slice(None, None, -1)})
        ds[y_coord] = np.ascontiguousarray(ds[y_coord].values)

    # Check the coords are all increasing now
    assert_values_unique_increasing(ds[x_coord].values, x_coord)
    assert_values_unique_increasing(ds[y_coord].values, y_coord)

    return ds


def get_xr_data_array_from_xr_dataset(ds: xr.Dataset) -> xr.DataArray:
    """Return underlying xr.DataArray from passed xr.Dataset.

    Checks only one variable is present and returns it as an xr.DataArray.

    Args:
        ds: xr.Dataset to extract xr.DataArray from
    """
    datavars = list(ds.data_vars)
    if len(datavars) != 1:
        raise ValueError("Cannot open as xr.DataArray: dataset contains multiple variables")
    return ds[datavars[0]]


def open_zarr_paths(
    zarr_path: str | list[str],
    time_dim: str,
    backend: Literal["dask", "tensorstore"],
    public: bool = False,
) -> xr.Dataset:
    """Opens the NWP data.

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension
        public: Whether the data is public or private. Only available for the dask backend.
        backend: The xarray backend to use.

    Returns:
        The opened Xarray Dataset
    """
    if backend not in ["dask", "tensorstore"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'dask' and 'tensorstore'.",
        )

    if public and backend == "tensorstore":
        raise ValueError("Public data is only supported with the 'dask' backend.")

    if backend == "tensorstore":
        ds = _tensorstore_open_zarr_paths(zarr_path, time_dim)

    elif backend == "dask":
        ds = _dask_open_zarr_paths(zarr_path, time_dim, public)

    return ds


def _dask_open_zarr_paths(zarr_path: str | list[str], time_dim: str, public: bool) -> xr.Dataset:
    general_kwargs = {
        "engine": "zarr",
        "chunks": "auto",
        "decode_timedelta": True,
    }

    if public:
        # note this only works for s3 zarr paths at the moment
        general_kwargs["storage_options"] = {"anon": True}

    if isinstance(zarr_path, list | tuple) or "*" in str(zarr_path):  # Multi-file dataset
        ds = xr.open_mfdataset(
            zarr_path,
            concat_dim=time_dim,
            combine="nested",
            coords="different",
            compat="no_conflicts",
            **general_kwargs,
        ).sortby(time_dim)
    else:
        ds = xr.open_dataset(
            zarr_path,
            consolidated=True,
            mode="r",
            **general_kwargs,
        )
    return ds


def _tensorstore_open_zarr_paths(zarr_path: str | list[str], time_dim: str) -> xr.Dataset:

    if "*" in str(zarr_path):
        zarr_path = sorted(glob(zarr_path))

    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim=time_dim, data_source="nwp").sortby(time_dim)
    else:
        ds = open_zarr(zarr_path)
    return ds

