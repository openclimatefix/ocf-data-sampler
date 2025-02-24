"""Save functions for the configuration model.

This module provides functionality to save configuration objects to YAML files,
supporting local and cloud storage locations.
"""

import json
import os

import fsspec
import yaml

from ocf_data_sampler.config import Configuration


def save_yaml_configuration(configuration: Configuration, filename: str) -> None:
    """Save a configuration object to a YAML file.

    Args:
        configuration: Configuration object containing the settings to save
        filename: Destination path for the YAML file. Can be a local path or
                 cloud storage URL (e.g., 'gs://', 's3://'). For local paths,
                 absolute paths are recommended.
    """
    if os.path.exists(filename):
        raise FileExistsError(f"File already exists: {filename}")

    # Serialize configuration to JSON-compatible dictionary
    config_dict = json.loads(configuration.model_dump_json())

    with fsspec.open(filename, mode="w") as yaml_file:
        yaml.safe_dump(config_dict, yaml_file, default_flow_style=False)
