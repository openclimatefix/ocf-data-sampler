"""Utility functions for the NWP data processing."""

import xarray as xr

from ocf_data_sampler.load.open_tensorstore_zarrs import open_zarr, open_zarrs

def open_zarr_paths(
    zarr_path: str | list[str], 
    time_dim: str = "init_time", 
    public: bool = False,
) -> xr.Dataset:
    """Opens the NWP data.

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension
        public: Whether the data is public or private

    Returns:
        The opened Xarray Dataset
    """

    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim=time_dim)
    else:
        ds = open_zarr(zarr_path)
    return ds
