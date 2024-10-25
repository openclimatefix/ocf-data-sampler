import tempfile

import pytest
from pydantic import ValidationError

from ocf_data_sampler.config import (
    load_yaml_configuration,
    Configuration,
    save_yaml_configuration
)


def test_default():
    """Test default pydantic class"""

    _ = Configuration()


def test_yaml_load_test_config(test_config_filename):
    """
    Test that yaml loading works for 'test_config.yaml'
    and fails for an empty .yaml file
    """

    # check we get an error if loading a file with no config
    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
        filename = fp.name

        # check that temp file can't be loaded
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(filename)

    # test can load test_config.yaml
    config = load_yaml_configuration(test_config_filename)

    assert isinstance(config, Configuration)


def test_yaml_save(test_config_filename):
    """
    Check configuration can be saved to a .yaml file
    """

    test_config = load_yaml_configuration(test_config_filename)

    with tempfile.NamedTemporaryFile(suffix=".yaml") as fp:
        filename = fp.name

        # save default config to file
        save_yaml_configuration(test_config, filename)

        # check the file can be loaded back
        tmp_config = load_yaml_configuration(filename)

        # check loaded configuration is the same as the one passed to save
        assert test_config == tmp_config


def test_extra_field():
    """
    Check an extra parameters in config causes error
    """

    configuration = Configuration()
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = Configuration(**configuration_dict)


def test_incorrect_forecast_minutes(test_config_filename):
    """
    Check a forecast length not divisible by time resolution causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].forecast_minutes = 1111
    with pytest.raises(Exception, match="duration must be divisible by time resolution"):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_history_minutes(test_config_filename):
    """
    Check a history length not divisible by time resolution causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].history_minutes = 1111
    with pytest.raises(Exception, match="duration must be divisible by time resolution"):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_nwp_provider(test_config_filename):
    """
    Check an unexpected nwp provider causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].nwp_provider = "unexpected_provider"
    with pytest.raises(Exception, match="NWP provider"):
        _ = Configuration(**configuration.model_dump())

def test_incorrect_dropout(test_config_filename):
    """
    Check a dropout timedelta over 0 causes error and 0 doesn't
    """

    configuration = load_yaml_configuration(test_config_filename)

    # check a positive number is not allowed
    configuration.input_data.nwp['ukv'].dropout_timedeltas_minutes = [120]
    with pytest.raises(Exception, match="Dropout timedeltas must be negative"):
        _ = Configuration(**configuration.model_dump())

    # check 0 is allowed
    configuration.input_data.nwp['ukv'].dropout_timedeltas_minutes = [0]
    _ = Configuration(**configuration.model_dump())

def test_incorrect_dropout_fraction(test_config_filename):
    """
    Check dropout fraction outside of range causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].dropout_fraction= 1.1
    with pytest.raises(Exception, match="Dropout fraction must be between 0 and 1"):
        _ = Configuration(**configuration.model_dump())

    configuration.input_data.nwp['ukv'].dropout_fraction= -0.1
    with pytest.raises(Exception, match="Dropout fraction must be between 0 and 1"):
        _ = Configuration(**configuration.model_dump())


def test_inconsistent_dropout_use(test_config_filename):
    """
    Check dropout fraction outside of range causes error
    """

    configuration = load_yaml_configuration(test_config_filename)
    configuration.input_data.satellite.dropout_fraction= 1.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = None

    with pytest.raises(ValueError, match="To dropout fraction > 0 requires a list of dropout timedeltas"):
        _ = Configuration(**configuration.model_dump())
    configuration.input_data.satellite.dropout_fraction= 0.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = [-120, -60]
    with pytest.raises(ValueError, match="To use dropout timedeltas dropout fraction should be > 0"):
        _ = Configuration(**configuration.model_dump())