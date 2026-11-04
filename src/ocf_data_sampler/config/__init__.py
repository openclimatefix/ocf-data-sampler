"""Configuration model."""

from ocf_data_sampler.config.io import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig

__all__ = ["PVNetDataConfig", "load_yaml_configuration", "save_yaml_configuration"]
