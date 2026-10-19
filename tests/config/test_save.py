"""Tests for configuration saving functionality."""

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import Channel, NWPChannel


def test_save_yaml_configuration_basic(tmp_path, config_filename):
    """Save an empty configuration object"""
    config = load_yaml_configuration(config_filename)
    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(config, filepath)
    assert filepath.exists()


def test_save_load_yaml_configuration(tmp_path, config_filename):
    """Make sure a saved configuration is the same after loading"""

    # Start with this config
    initial_config = load_yaml_configuration(config_filename)

    initial_config.satellite.channels = [
        Channel(name="z", mean=0, std=1), Channel(name="a", mean=1, std=2),
    ]
    initial_config.nwp["ukv"].channels = [
        NWPChannel(name="total", accumulated=True, mean=2, std=3),
        NWPChannel(name="temperature", mean=4, std=5),
    ]

    # Save it - then load and check it is identical
    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(initial_config, filepath)
    assert load_yaml_configuration(filepath) == initial_config
