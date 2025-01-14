"""Save functions for the configuration model.

This module provides functionality to save configuration objects to YAML files,
supporting local and cloud storage locations.

Example:
    from ocf_data_sampler.config import save_yaml_configuration
    saved_path = save_yaml_configuration(config, "config.yaml")
"""

import json

from pathlib import Path
from typing import Union

import fsspec
import yaml

from ocf_data_sampler.config import Configuration


def save_yaml_configuration(
    configuration: Configuration,
    filename: Union[str, Path],
) -> Path:
    """Save a configuration object to a YAML file.

    Args:
        configuration: Configuration object containing the settings to save
        filename: Destination path for the YAML file. Can be a local path or
                 cloud storage URL (e.g., 'gs://', 's3://'). For local paths,
                 absolute paths are recommended.

    Returns:
        Path: The path where the configuration was saved

    Raises:
        ValueError: If filename is None or if writing to the specified path fails
        TypeError: If the configuration cannot be serialized
    """
    if filename is None:
        raise ValueError("filename cannot be None")

    try:
        # Convert to absolute path if it's a relative path
        if isinstance(filename, (str, Path)) and not any(
            str(filename).startswith(prefix) for prefix in ('gs://', 's3://', '/')
        ):
            filename = Path.cwd() / filename

        filepath = Path(filename)

        # For local files, check if directory exists before proceeding
        if filepath.is_absolute():
            directory = filepath.parent
            if not directory.exists():
                raise ValueError("Directory does not exist")

        # Serialize configuration to JSON-compatible dictionary
        config_dict = json.loads(configuration.model_dump_json())

        # Save to YAML file using fsspec
        with fsspec.open(str(filepath), mode='w') as yaml_file:
            yaml.safe_dump(config_dict, yaml_file, default_flow_style=False)

        return filepath

    except json.JSONDecodeError as e:
        raise TypeError(f"Failed to serialize configuration: {str(e)}") from e
    except PermissionError as e:
        raise ValueError(f"Permission denied when writing to {filename}") from e
    except (IOError, OSError) as e:
        raise ValueError(f"Failed to write configuration to {filename}: {str(e)}") from e