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
    """Helper function to open a zarr from either local or GCP path.

    The local or GCP path may contain wildcard matching (*)

    Args:
        zarr_path: Path to zarr file
    """

    # These kwargs are used if zarr path contains "*"
    openmf_kwargs = dict(
        engine="zarr",
        concat_dim="time",
        combine="nested",
        chunks="auto",
        join="override",
    )

    # Need to generate list of files if using GCP bucket storage
    if "gs://" in str(zarr_path) and "*" in str(zarr_path):
        result_string = subprocess.run(
            f"gsutil ls -d {zarr_path}".split(" "), stdout=subprocess.PIPE
        ).stdout.decode("utf-8")
        files = result_string.splitlines()

        ds = xr.open_mfdataset(files, **openmf_kwargs)

    elif "*" in str(zarr_path):  # Multi-file dataset
        ds = xr.open_mfdataset(zarr_path, **openmf_kwargs)
    else:
        ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")
    ds = ds.drop_duplicates("time").sortby("time")

    return ds


def open_sat_data(zarr_path: Path | str | list[Path] | list[str]) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Args:
      zarr_path: Cloud URL or local path pattern, or list of these. If GCS URL, it must start with
          'gs://'.

    Example:
        With wild cards and GCS path:
        ```
        zarr_paths = [
            "gs://bucket/2020_nonhrv_split_*.zarr",
            "gs://bucket/2019_nonhrv_split_*.zarr",
        ]
        ds = open_sat_data(zarr_paths)
        ```
        Without wild cards and with local path:
        ```
        zarr_paths = [
            "/data/2020_nonhrv.zarr",
            "/data/2019_nonhrv.zarr",
        ]
        ds = open_sat_data(zarr_paths)
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

    # Rename
    ds = ds.rename(
        {
            "variable": "channel",
            "time": "time_utc",
        }
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")

    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
