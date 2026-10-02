"""Load and save configuration objects as local YAML files."""

import json

import yaml

from ocf_data_sampler.config.model import PVNetDataConfig


def load_yaml_configuration(filename: str) -> PVNetDataConfig:
    """Load a yaml file which has a configuration in it.

    Args:
        filename: Local path to the YAML file to load.

    Returns: pydantic class
    """
    with open(filename, encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)

    return PVNetDataConfig(**configuration)


def save_yaml_configuration(configuration: PVNetDataConfig, filename: str) -> None:
    """Save a configuration object to a YAML file.

    Args:
        configuration: PVNetDataConfig object containing the settings to save
        filename: Local destination path for the YAML file.

    Raises:
        FileExistsError: If the destination already exists.
    """
    # Serialize configuration to JSON-compatible dictionary
    config_dict = json.loads(configuration.model_dump_json())

    with open(filename, mode="x", encoding="utf-8") as yaml_file:
        yaml.safe_dump(config_dict, yaml_file, default_flow_style=False)
