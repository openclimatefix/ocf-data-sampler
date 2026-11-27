"""Shared infrastructure."""

from ocf_data_sampler.common.xr_tensorstore import concat_tensorstore, open_zarr_paths

__all__ = ["concat_tensorstore", "open_zarr_paths"]
