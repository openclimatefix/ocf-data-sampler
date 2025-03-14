"""Utility functions for working with xarray objects."""

import numpy as np
import pandas as pd
import xarray as xr


def check_time_unique_increasing(datetimes: xr.DataArray) -> None:
    """Check that the time dimension is unique and increasing."""
    time = pd.DatetimeIndex(datetimes)
    if not (time.is_unique and time.is_monotonic_increasing):
        raise ValueError("Time dimension must be unique and monotonically increasing")


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
    if not (ds[x_coord].diff(dim=x_coord) > 0).all():
        raise ValueError(f"'{x_coord}' coordinate must be increasing")
    if not (ds[y_coord].diff(dim=y_coord) > 0).all():
        raise ValueError(f"'{y_coord}' coordinate must be increasing")

    return ds


def get_xr_data_array_from_xr_dataset(ds: xr.Dataset) -> xr.DataArray:
    """Return underlying xr.DataArray from passed xr.Dataset.

    Checks only one variable is present and returns it as an xr.DataArray.

    Args:
        ds: xr.Dataset to extract xr.DataArray from
    """
    datavars = list(ds.var())
    if len(datavars) != 1:
        raise ValueError("Cannot open as xr.DataArray: dataset contains multiple variables")
    return ds[datavars[0]]
