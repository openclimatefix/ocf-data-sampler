"""Save functions for the configuration model.

Example:

    from ocf_data_sampler.config import save_yaml_configuration
    configuration = save_yaml_configuration(config, filename)
"""

import json

import fsspec
import yaml
from pathy import Pathy

from ocf_data_sampler.config import Configuration


def save_yaml_configuration(
    configuration: Configuration, filename: str | Pathy
):
    """
    Save a local yaml file which has the configuration in it.

    If `filename` is None then saves to configuration.output_data.filepath / configuration.yaml.

    Will save to GCP, AWS, or local, depending on the protocol suffix of filepath.
    """
    # make a dictionary from the configuration,
    # Note that we make the object json'able first, so that it can be saved to a yaml file
    d = json.loads(configuration.model_dump_json())
    if filename is None:
        filename = Pathy(configuration.output_data.filepath) / "configuration.yaml"

    # save to a yaml file
    with fsspec.open(filename, "w") as yaml_file:
        yaml.safe_dump(d, yaml_file, default_flow_style=False)
