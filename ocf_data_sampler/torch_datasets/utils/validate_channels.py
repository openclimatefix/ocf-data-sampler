import logging
import xarray as xr


def _validate_channels(config_channels, means_channels, stds_channels, provider_name=None, is_satellite=False):
    """
    Common validation logic for both NWP and satellite channels

    Args:
        config_channels: Set of channels from config
        means_channels: Set of channels from means constants
        stds_channels: Set of channels from stds constants
        provider_name: Name of NWP provider if applicable
        is_satellite: Boolean indicating if checking satellite channels

    Raises:
        ValueError: If there's a mismatch between config channels and specified normalisation constants
    """

    prefix = "satellite channels" if is_satellite else f"channels for {provider_name}"
    constant_prefix = "RSS" if is_satellite else "NWP"

    # Find missing channels in means
    missing_in_means = config_channels - means_channels
    if missing_in_means:
        raise ValueError(
            f"The following {prefix} are missing in {constant_prefix}_MEANS: {missing_in_means}"
        )
            
    # Find missing channels in stds
    missing_in_stds = config_channels - stds_channels
    if missing_in_stds:
        raise ValueError(
            f"The following {prefix} are missing in {constant_prefix}_STDS: {missing_in_stds}"
        )
            
    # Check if any extra channels in constants that are not in config
    extra_in_means = means_channels - config_channels
    extra_in_stds = stds_channels - config_channels
    
    if extra_in_means or extra_in_stds:
        warning_prefix = "The following"
        warning_suffix = (
            "satellite channels exist in normalisation constants but are not used in the config"
            if is_satellite
            else f"channels exist in normalisation constants but are not used in the config for {provider_name}"
        )
        logging.warning(
            f"{warning_prefix} {warning_suffix}: "
            f"In means: {extra_in_means}, In stds: {extra_in_stds}"
        )


def validate_nwp_channels(config, nwp_means, nwp_stds):
    """
    Validates that all NWP channels specified in config have corresponding
    normalisation constants.

    Args:
        config: Configuration object containing NWP channel specifications
        nwp_means: Dictionary of NWP means per provider
        nwp_stds: Dictionary of NWP standard deviations per provider

    Raises:
        ValueError: If there's a mismatch between config channels and normalisation constants
    """
    if not hasattr(config.input_data, 'nwp'):
        return
    
    for provider_key, provider_config in config.input_data.nwp.items():
        provider_name = provider_config.provider
        
        if provider_name not in nwp_means or provider_name not in nwp_stds:
            raise ValueError(f"Provider {provider_name} not found in normalisation constants")
            
        _validate_channels(
            config_channels=set(provider_config.channels),
            means_channels=set(nwp_means[provider_name].channel.values),
            stds_channels=set(nwp_stds[provider_name].channel.values),
            provider_name=provider_name
        )


def validate_sat_channels(config, rss_means, rss_stds):
    """
    Validates that all satellite channels specified in config have corresponding
    normalisation constants.

    Args:
        config: Configuration object containing satellite channel specifications
        rss_means: Dictionary or DataArray of satellite means
        rss_stds: Dictionary or DataArray of satellite standard deviations

    Raises:
        ValueError: If there's a mismatch between config channels and normalisation constants
    """
    if not hasattr(config.input_data, 'satellite') or config.input_data.satellite is None:
        return
        
    _validate_channels(
        config_channels=set(config.input_data.satellite.channels),
        means_channels=set(rss_means.channel.values),
        stds_channels=set(rss_stds.channel.values),
        is_satellite=True
    )
