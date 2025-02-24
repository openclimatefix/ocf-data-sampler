"""Tests for configuration saving functionality."""

import os

from ocf_data_sampler.config import Configuration, load_yaml_configuration, save_yaml_configuration


def test_save_yaml_configuration_basic(tmp_path):
    """Save an empty configuration object"""
    config = Configuration()

    filepath = f"{tmp_path}/config.yaml"
    save_yaml_configuration(config, filepath)

    assert os.path.exists(filepath)


def test_save_load_yaml_configuration(tmp_path, test_config_filename):
    """Make sure a saved configuration is the same after loading"""

    # Start with this config
    initial_config = load_yaml_configuration(test_config_filename)

    # Save it
    filepath = f"{tmp_path}/config.yaml"
    save_yaml_configuration(initial_config, filepath)

    # Load it and check it is still the same
    loaded_config = load_yaml_configuration(filepath)
    assert loaded_config == initial_config
