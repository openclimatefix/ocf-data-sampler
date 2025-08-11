"""Utility functions for the NWP data processing."""

from glob import glob

import xarray as xr

from ocf_data_sampler.load.open_xarray_tensorstore import open_zarr, open_zarrs


def open_zarr_paths(
    zarr_path: str | list[str],
    time_dim: str = "init_time",
    public: bool = False,
    backend: str = "dask",
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
        ds = _tensostore_open_zarr_paths(zarr_path, time_dim)

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


def _tensostore_open_zarr_paths(zarr_path: str | list[str], time_dim: str) -> xr.Dataset:

    if "*" in str(zarr_path):
        zarr_path = sorted(glob(zarr_path))

    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim=time_dim).sortby(time_dim)
    else:
        ds = open_zarr(zarr_path)
    return ds

