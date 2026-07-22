"""Helpers that conform freshly opened data to package conventions."""

from collections.abc import Mapping

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing


def _is_expected_dtype(actual_dtype: np.dtype, expected_dtype: type[np.generic]) -> bool:
    """Return whether the coordinate dtype matches the expected dtype contract."""
    if np.issubdtype(actual_dtype, expected_dtype):
        return True

    return expected_dtype is np.str_ and actual_dtype.kind in {"U", "T"}


def validate_coords(
    data: xr.Dataset | xr.DataArray,
    expected_dtypes: Mapping[str, type[np.generic]],
    source: str,
) -> None:
    """Validate required coordinate presence, dimensionality, and dtypes.

    Args:
        data: Xarray object containing the coordinates.
        expected_dtypes: Mapping from coordinate names to expected NumPy dtype classes.
        source: Description of the data source used in validation errors.
    """
    for coord, expected_dtype in expected_dtypes.items():
        if coord not in data.coords:
            raise ValueError(f"Expected coordinate {coord!r} missing from {source}")

        if (ndim := data[coord].ndim) != 1:
            raise ValueError(
                f"Coordinate {coord!r} in {source} should be 1D, not {ndim}D",
            )

        actual_dtype = data[coord].dtype
        if not _is_expected_dtype(actual_dtype, expected_dtype):
            raise TypeError(
                f"Coordinate {coord!r} in {source} should be "
                f"{expected_dtype.__name__}, not {actual_dtype.name}",
            )


def make_spatial_coords_increasing(
    ds: xr.Dataset | xr.DataArray,
    x_coord: str,
    y_coord: str,
) -> xr.Dataset | xr.DataArray:
    """Make sure the spatial coordinates are in increasing order.

    Args:
        ds: Xarray Dataset
        x_coord: Name of the x coordinate
        y_coord: Name of the y coordinate
    """
    # Make sure the coords are in increasing order
    if ds[x_coord][0] > ds[x_coord][-1]:
        ds = ds.isel({x_coord: slice(None, None, -1)})
        # Below we copy the coord values so we don't have numpy array with negative strides
        # Numpy arrays with negative strides cannot be converted to torch Tensor
        ds[x_coord] = np.ascontiguousarray(ds[x_coord].values)
    if ds[y_coord][0] > ds[y_coord][-1]:
        ds = ds.isel({y_coord: slice(None, None, -1)})
        ds[y_coord] = np.ascontiguousarray(ds[y_coord].values)

    # Check the coords are all increasing now
    assert_values_unique_increasing(ds[x_coord].values, x_coord)
    assert_values_unique_increasing(ds[y_coord].values, y_coord)

    return ds


def extract_single_data_array(ds: xr.Dataset, promote_attrs: bool = True) -> xr.DataArray:
    """Return underlying xr.DataArray from passed xr.Dataset.

    Checks only one variable is present and returns it as an xr.DataArray.

    Args:
        ds: xr.Dataset to extract xr.DataArray from
        promote_attrs: Whether to promote dataset attributes to the data array
    """
    datavars = list(ds.data_vars)
    if len(datavars) != 1:
        raise ValueError(
            f"Cannot extract a single DataArray: dataset contains variables {datavars}",
        )
    da = ds[datavars[0]]
    if promote_attrs:
        da.attrs.update(ds.attrs)
    return da
