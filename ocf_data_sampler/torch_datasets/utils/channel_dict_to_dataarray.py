"""Utility function for converting channel dictionaries to xarray DataArrays."""

import xarray as xr


def channel_dict_to_dataarray(channel_dict: dict[str, float]) -> xr.DataArray:
    """Converts a dictionary of channel values to a DataArray.

    Args:
        channel_dict: Dictionary mapping channel names (str) to their values (float).

    Returns:
        xr.DataArray: A 1D DataArray with channels as coordinates.
    """
    return xr.DataArray(
        list(channel_dict.values()),
        coords={"channel": list(channel_dict.keys())},
    )
