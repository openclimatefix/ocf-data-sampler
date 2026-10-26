"""Tests for loading and saving YAML configurations."""

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig


def test_save_yaml_configuration_basic(tmp_path, config_filename):
    """Save an empty configuration object"""
    config = load_yaml_configuration(config_filename)
    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(config, filepath)
    assert filepath.exists()


def test_load_yaml_configuration(config_filename):
    """Loading a YAML configuration returns a PVNetDataConfig."""
    assert isinstance(load_yaml_configuration(config_filename), PVNetDataConfig)


def test_save_load_yaml_configuration(tmp_path, config_filename):
    """Make sure a saved configuration is the same after loading"""

    initial_config = load_yaml_configuration(config_filename)

    filepath = tmp_path / "config.yaml"
    save_yaml_configuration(initial_config, filepath)
    assert load_yaml_configuration(filepath) == initial_config
