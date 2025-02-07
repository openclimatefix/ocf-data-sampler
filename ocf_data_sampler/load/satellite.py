"""Satellite loader"""

import subprocess
from pathlib import Path

import xarray as xr
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    make_spatial_coords_increasing,
    get_xr_data_array_from_xr_dataset
)


def _get_single_sat_data(zarr_path: Path | str) -> xr.Dataset:
    """Helper function to open a Zarr from either a local or GCP path.

    Args:
        zarr_path: Path to Zarr file. Supports wildcard (*) only for local paths, not for GCP URLs (gs://).

    Returns:
        An xarray Dataset containing satellite data.

    Raises:
        ValueError: If wildcard (*) is used in a GCP (gs://) path.
    """

    # These kwargs are used if the path contains "*"
    openmf_kwargs = dict(
        engine="zarr",
        concat_dim="time",
        combine="nested",
        chunks="auto",
        join="override",
    )

    # Raise an error if a wildcard is used in a GCP path
    if "gs://" in str(zarr_path) and "*" in str(zarr_path):
        raise ValueError("Wildcard (*) paths are not supported for GCP (gs://) URLs.")

    # Handle multi-file dataset for local paths
    if "*" in str(zarr_path):
        ds = xr.open_mfdataset(zarr_path, **openmf_kwargs)
    else:
        ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")

    # Ensure time is unique and sorted
    ds = ds.drop_duplicates("time").sortby("time")

    return ds


def open_sat_data(zarr_path: Path | str | list[Path] | list[str]) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Args:
      zarr_path: Cloud URL or local path pattern, or list of these.
                 If GCS URL, it must start with 'gs://'.
                 Wildcard (*) is only supported for local paths.

    Returns:
      An xarray DataArray containing satellite data.

    Example:
        Opening multiple local paths with wildcards:
        ```
        ds = open_sat_data("/data/*.zarr")
        ```
        Opening GCP paths without wildcards:
        ```
        ds = open_sat_data("gs://bucket/2020_nonhrv.zarr")
        ```
    """

    # Open the data
    if isinstance(zarr_path, (list, tuple)):
        ds = xr.combine_nested(
            [_get_single_sat_data(path) for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        ds = _get_single_sat_data(zarr_path)

    # Rename dimensions
    ds = ds.rename(
        {
            "variable": "channel",
            "time": "time_utc",
        }
    )

    # Check timestamps
    check_time_unique_increasing(ds.time_utc)

    # Ensure spatial coordinates are sorted
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")

    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")

    return get_xr_data_array_from_xr_dataset(ds)
