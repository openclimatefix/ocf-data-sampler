"""Utility functions for the NWP data processing."""

import xarray as xr

from ocf_data_sampler.config.model import NWP


def open_zarr_paths(nwp_config: NWP, time_dim: str = "init_time") -> xr.Dataset:
    """Opens the NWP data.

    Args:
        nwp_config: NWP configuration object
        time_dim: Name of the time dimension

    Returns:
        The opened Xarray Dataset
    """

    general_kwargs = {
        "engine": "zarr",
        "chunks": "auto",
        "decode_timedelta": True,
    }

    if nwp_config.public:
        # note this only works for s3 zarr paths at the moment
        general_kwargs["storage_options"] = {"anon": True}

    zarr_path = nwp_config.zarr_path
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
