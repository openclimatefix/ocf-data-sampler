"""Converts a dictionary of channel values to a DataArray."""

import xarray as xr


def channel_dict_to_dataarray(channel_dict: dict[str, float]) -> xr.DataArray:
    """Converts a dictionary of channel values to a DataArray."""
    return xr.DataArray(
        list(channel_dict.values()),
        coords={"channel": list(channel_dict.keys())},
    )
