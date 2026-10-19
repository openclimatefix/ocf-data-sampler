"""Takes the diff along the step axis for a given set of channels."""

import numpy as np

from ocf_data_sampler.common.types import TArray


def diff_channels(da: TArray, accum_channels: list[str]) -> TArray:
    """Difference selected channels along step without modifying the input.

    The returned array is sliced to one step shorter to allow for differencing the
    configured channels. Differenced channel names are prefixed with ``diff_``.
    Differences retain the earlier step's label.

    Args:
        da: The DataArray to slice from
        accum_channels: Channels which are accumulated and need to be differenced
    """
    if da.dims[:2] != ("step", "channel"):
        raise ValueError("This function assumes the first two dimensions are step then channel")

    all_channels = da["channel"].values
    accum_channel_inds = [i for i, c in enumerate(all_channels) if c in accum_channels]

    da_out = da.isel(step=slice(0, -1))
    all_values = da.values

    out_values = all_values[:-1].copy()
    out_values[:, accum_channel_inds] = np.diff(all_values[:, accum_channel_inds], axis=0)
    da_out.data = out_values

    new_channels = np.array([
        f"diff_{channel}" if channel in accum_channels else channel
        for channel in all_channels
    ])
    da_out.coords["channel"] = (("channel",), new_channels)

    return da_out
