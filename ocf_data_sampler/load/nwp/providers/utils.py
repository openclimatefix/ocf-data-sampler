"""Utility functions for the NWP data processing."""

import xarray as xr


def open_zarr_paths(
    zarr_path: str | list[str], time_dim: str = "init_time", public: bool = False,
) -> xr.Dataset:
    """Opens the NWP data.

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension
        public: Whether the data is public or private

    Returns:
        The opened Xarray Dataset
    """
    general_kwargs = {
        "engine": "zarr",
        "chunks": "auto",
        "decode_timedelta": True,
    }

    if public:
        # note this only works for s3 zarr paths at the moment
        general_kwargs["storage_options"] = {"anon": True}

    if type(zarr_path) in [list, tuple] or "*" in str(zarr_path):  # Multi-file dataset
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
