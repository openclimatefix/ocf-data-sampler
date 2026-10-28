"""Takes the diff along the step axis for a given set of channels."""

import numpy as np

from ocf_data_sampler.common.types import TArray


def diff_channels_along_step(da: TArray, channels: list[str]) -> TArray:
    """Difference selected channels along step without modifying the input.

    The returned array is sliced to one step shorter to allow for differencing the
    configured channels. Differenced channel names are prefixed with ``diff_``.
    Differences retain the earlier step's label.

    Args:
        da: The DataArray to slice from
        channels: Names of the channels to difference along the step axis.
    """
    if da.dims[:2] != ("step", "channel"):
        raise ValueError("This function assumes the first two dimensions are step then channel")

    if len(channels)==0:
        raise ValueError("At least one channel must be requested")

    all_channels = da["channel"].values.tolist()

    if set(channels) - set(all_channels):
        raise ValueError(f"{channels=} not all available in {all_channels=}")

    channel_indices = [all_channels.index(c) for c in channels]

    da_out = da.isel(step=slice(0, -1))
    all_values = da.values

    out_values = all_values[:-1].copy()
    out_values[:, channel_indices] = np.diff(all_values[:, channel_indices], axis=0)
    da_out.data = out_values

    new_channels = np.array([
        f"diff_{channel}" if channel in channels else channel
        for channel in all_channels
    ])
    da_out.coords["channel"] = (("channel",), new_channels)

    return da_out
