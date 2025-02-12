import logging
import xarray as xr

def validate_channels(
    data_channels: set,
    means_channels: set,
    stds_channels: set,
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
    # Find missing channels in means
    missing_in_means = data_channels - means_channels
    if missing_in_means:
        raise ValueError(
            f"The following channels for {source_name} are missing in normalisation means: "
            f"{missing_in_means}"
        )
            
    # Find missing channels in stds
    missing_in_stds = data_channels - stds_channels
    if missing_in_stds:
        raise ValueError(
            f"The following channels for {source_name} are missing in normalisation stds: "
            f"{missing_in_stds}"
        )
            
    # Check if any extra channels in constants that are not in data
    extra_in_means = means_channels - data_channels
    extra_in_stds = stds_channels - data_channels
    
    if extra_in_means or extra_in_stds:
        logging.warning(
            f"The following channels exist in normalisation constants but are not used "
            f"for {source_name}: In means: {extra_in_means}, In stds: {extra_in_stds}"
        )
