"""Takes the diff along the step axis for a given set of channels."""

import numpy as np
import xarray as xr


def diff_channels(da: xr.DataArray, accum_channels: list[str]) -> xr.DataArray:
    """Perform in-place diff of the given channels of the DataArray in the steps dimension.

    Args:
        da: The DataArray to slice from
        accum_channels: Channels which are accumulated and need to be differenced
    """
    if da.dims[:2] != ("step", "channel"):
        raise ValueError("This function assumes the first two dimensions are step then channel")

    all_channels = da.channel.values
    accum_channel_inds = [i for i, c in enumerate(all_channels) if c in accum_channels]

    # Make a copy of the values to avoid changing the underlying numpy array
    vals = da.values.copy()
    vals[:-1, accum_channel_inds] = np.diff(vals[:, accum_channel_inds], axis=0)
    da.values = vals

    return da.isel(step=slice(0, -1))
