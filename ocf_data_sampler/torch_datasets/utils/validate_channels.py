import xarray as xr

def validate_channels(
    data_channels: list,
    means_channels: list,
    stds_channels: list,
    source_name: str | None = None
) -> None:
    """
    Validates that all channels in data have corresponding normalisation constants.

    Args:
        data_channels: Set of channels from the data
        means_channels: Set of channels from means constants
        stds_channels: Set of channels from stds constants
        source_name: Name of data source (e.g., 'ecmwf', 'satellite') for error messages

    Raises:
        ValueError: If there's a mismatch between data channels and normalisation constants
    """

    data_set = set(data_channels)
    means_set = set(means_channels)
    stds_set = set(stds_channels)

    # Find missing channels in means
    missing_in_means = data_set - means_set
    if missing_in_means:
        raise ValueError(
            f"The following channels for {source_name} are missing in normalisation means: "
            f"{missing_in_means}"
        )
            
    # Find missing channels in stds
    missing_in_stds = data_set - stds_set
    if missing_in_stds:
        raise ValueError(
            f"The following channels for {source_name} are missing in normalisation stds: "
            f"{missing_in_stds}"
        )
