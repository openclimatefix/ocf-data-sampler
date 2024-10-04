"""Loading configuration functions.

Example:

    from ocf_data_sampler.config import load_yaml_configuration
    configuration = load_yaml_configuration(filename)
"""

import fsspec
from pathy import Pathy
from pyaml_env import parse_config

from ocf_data_sampler.config import Configuration


def load_yaml_configuration(filename: str | Pathy) -> Configuration:
    """
    Load a yaml file which has a configuration in it

    Args:
        filename: the file name that you want to load.  Will load from local, AWS, or GCP
            depending on the protocol suffix (e.g. 's3://bucket/config.yaml').

    Returns:pydantic class

    """
    # load the file to a dictionary
    with fsspec.open(filename, mode="r") as stream:
        configuration = parse_config(data=stream)
        # this means we can load ENVs in the yaml file
    # turn into pydantic class
    configuration = Configuration(**configuration)
    return configuration
