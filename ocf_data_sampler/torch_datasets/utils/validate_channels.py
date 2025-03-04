"""Functions for checking that normalisation statistics exist for the data channels requested."""

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS, RSS_MEAN, RSS_STD


def validate_channels(
    data_channels: list,
    means_channels: list,
    stds_channels: list,
    source_name: str | None = None,
) -> None:
    """Validates that all channels in data have corresponding normalisation constants.

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
            f"{missing_in_means}",
        )

    # Find missing channels in stds
    missing_in_stds = data_set - stds_set
    if missing_in_stds:
        raise ValueError(
            f"The following channels for {source_name} are missing in normalisation stds: "
            f"{missing_in_stds}",
        )


def validate_nwp_channels(config: Configuration) -> None:
    """Validate that NWP channels in config have corresponding normalisation constants.

    Args:
        config: Configuration object containing NWP channel information

    Raises:
        ValueError: If there's a mismatch between configured NWP channels
        and normalisation constants
    """
    if hasattr(config.input_data, "nwp") and (
        config.input_data.nwp is not None
        ):
        for _, nwp_config in config.input_data.nwp.items():
            provider = nwp_config.provider
            validate_channels(
                data_channels=nwp_config.channels,
                means_channels=NWP_MEANS[provider].channel.values,
                stds_channels=NWP_STDS[provider].channel.values,
                source_name=provider,
            )


def validate_satellite_channels(config: Configuration) -> None:
    """Validate that satellite channels in config have corresponding normalisation constants.

    Args:
        config: Configuration object containing satellite channel information

    Raises:
        ValueError: If there's a mismatch between configured satellite channels
        and normalisation constants
    """
    if hasattr(config.input_data, "satellite") and (
        config.input_data.satellite is not None
        ):
        validate_channels(
            data_channels=config.input_data.satellite.channels,
            means_channels=RSS_MEAN.channel.values,
            stds_channels=RSS_STD.channel.values,
            source_name="satellite",
        )
