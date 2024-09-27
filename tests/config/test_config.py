"""Test config."""

import os
import tempfile

import pytest
from pydantic import ValidationError

from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.config.model import Configuration
from ocf_data_sampler.config.save import save_yaml_configuration


def test_default():
    """
    Test default pydantic class
    """

    _ = Configuration()


def test_yaml_load_on_premises(top_test_directory):
    """Test that yaml loading works for 'on_premises.yaml'"""

    filename = f"{top_test_directory}/test_data/configs/on_premises.yaml"

    config = load_yaml_configuration(filename)

    assert isinstance(config, Configuration)


def test_yaml_save():
    """
    Check a configuration can be saved to a yaml file
    """

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
        filename = fp.name

        # check that temp file cant be loaded
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(filename)

        # save default config to file
        save_yaml_configuration(Configuration(), filename)

        # check the file can be loaded
        _ = load_yaml_configuration(filename)


def test_yaml_load_env(test_config_filename):
    """
    Check a configuration can be loaded with an env var
    """

    os.environ["PATH"] = "example_path"

    # check the file can be loaded
    config_load = load_yaml_configuration(test_config_filename)

    assert "example_path" in config_load.general.description


def test_extra_field():
    """
    Check a extra parameters in config causes error
    """

    configuration = Configuration()
    configuration_dict = configuration.dict()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError):
        _ = Configuration(**configuration_dict)


def test_incorrect_forecast_minutes(top_test_directory):
    """
    Check a forecast length not divisible by time resolution causes error
    """

    filename = f"{top_test_directory}/test_data/configs/on_premises.yaml"
    configuration = load_yaml_configuration(filename)

    configuration.input_data.nwp['ukv'].forecast_minutes = 1111
    with pytest.raises(Exception):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_history_minutes(top_test_directory):
    """
    Check a history length not divisible by time resolution causes error
    """

    filename = f"{top_test_directory}/test_data/configs/on_premises.yaml"
    configuration = load_yaml_configuration(filename)

    configuration.input_data.nwp['ukv'].history_minutes = 1111
    with pytest.raises(Exception):
        _ = Configuration(**configuration.dict())
