"""Satellite loader"""

import logging
import subprocess
from pathlib import Path

import pandas as pd
import xarray as xr
from ocf_blosc2 import Blosc2  # noqa: F401


_log = logging.getLogger(__name__)


def _get_single_sat_data(zarr_path: Path | str) -> xr.DataArray:
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

    if isinstance(zarr_path, (list, tuple)):
        message_files_list = "\n - " + "\n - ".join([str(s) for s in zarr_path])
        _log.info(f"Opening satellite data: {message_files_list}")
        ds = xr.combine_nested(
            [_get_single_sat_data(path) for path in zarr_path],
            concat_dim="time",
            combine_attrs="override",
            join="override",
        )
    else:
        _log.info(f"Opening satellite data: {zarr_path}")
        ds = _get_single_sat_data(zarr_path)


    ds = ds.rename({"variable": "channel"})

    # Rename coords to be more explicit about exactly what some coordinates hold:
    # Note that `rename` renames *both* the coordinates and dimensions, and keeps
    # the connection between the dims and coordinates, so we don't have to manually
    # use `data_array.set_index()`.
    ds = ds.rename({"time": "time_utc"})

    # Flip coordinates to top-left first
    if ds.y_geostationary[0] < ds.y_geostationary[-1]:
        ds = ds.isel(y_geostationary=slice(None, None, -1))
    if ds.x_geostationary[0] > ds.x_geostationary[-1]:
        ds = ds.isel(x_geostationary=slice(None, None, -1))

    # Ensure the y and x coords are in the right order (top-left first):
    assert ds.y_geostationary[0] > ds.y_geostationary[-1]
    assert ds.x_geostationary[0] < ds.x_geostationary[-1]

    ds = ds.transpose("time_utc", "channel", "y_geostationary", "x_geostationary")

    # Sanity checks!
    datetime_index = pd.DatetimeIndex(ds.time_utc)
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing

    # Return DataArray
    return ds["data"]