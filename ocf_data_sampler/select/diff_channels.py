"""Takes the diff along the step axis for a given set of channels."""

import numpy as np
import xarray as xr


def diff_channels(da: xr.DataArray, accum_channels: list[str]) -> xr.DataArray:
    """Diff the given channels of the DataArray in-place in the steps dimension.

    Args:
        da: The DataArray to slice from
        accum_channels: Channels which are accumulated and need to be differenced
    """
    if da.dims[:2] != ("step", "channel"):
        raise ValueError("This function assumes the first two dimensions are step then channel")

    all_channels = da.channel.values
    accum_channel_inds = [i for i, c in enumerate(all_channels) if c in accum_channels]

    da.values[:-1, accum_channel_inds] = np.diff(da.values[:, accum_channel_inds], axis=0)

    return da.isel(step=slice(0, -1))
