"""Tests for configuration saving functionality."""

import os

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration


def test_save_yaml_configuration_basic(tmp_path, test_config_gsp_path):
    """Save an empty configuration object"""
    config = load_yaml_configuration(test_config_gsp_path)
    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(config, filepath)
    assert filepath.exists()


def test_save_load_yaml_configuration(tmp_path, test_config_filename):
    """Make sure a saved configuration is the same after loading"""

    # Start with this config
    initial_config = load_yaml_configuration(test_config_filename)

    # Save it - then load and check it is identical
    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(initial_config, filepath)
    assert load_yaml_configuration(filepath) == initial_config
