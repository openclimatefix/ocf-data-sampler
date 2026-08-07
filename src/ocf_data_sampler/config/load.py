"""Load configuration from a yaml file."""

import fsspec
from pyaml_env import parse_config

from ocf_data_sampler.config.model import PVNetDataConfig


def load_yaml_configuration(filename: str) -> PVNetDataConfig:
    """Load a yaml file which has a configuration in it.

    Args:
        filename: the yaml file name that you want to load.  Will load from local, AWS, or GCP
            depending on the protocol suffix (e.g. 's3://bucket/config.yaml').

    Returns: pydantic class

    """
    with fsspec.open(filename, mode="r") as stream:
        configuration = parse_config(data=stream)

    return PVNetDataConfig(**configuration)
