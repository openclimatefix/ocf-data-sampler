import tempfile

import pytest
from pydantic import ValidationError
from pathlib import Path
from ocf_data_sampler.config import (
    load_yaml_configuration,
    Configuration,
    save_yaml_configuration
)


def test_default_configuration():
    """Test default pydantic class"""

    _ = Configuration()


def test_load_yaml_configuration(test_config_filename):
    """
    Test that yaml loading works for 'test_config.yaml'
    and fails for an empty .yaml file
    """
    # Create temporary directory instead of file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path for empty file
        empty_file = Path(temp_dir) / "empty.yaml"
        
        # Create an empty file
        empty_file.touch()
        
        # Test loading empty file
        with pytest.raises(TypeError):
            _ = load_yaml_configuration(str(empty_file))

def test_yaml_save(test_config_filename):
    """
    Check configuration can be saved to a .yaml file
    """
    test_config = load_yaml_configuration(test_config_filename)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path for config file
        config_path = Path(temp_dir) / "test_config.yaml"
        
        # Save configuration
        saved_path = save_yaml_configuration(test_config, config_path)
        
        # Verify file exists
        assert saved_path.exists()
        
        # Test loading saved configuration
        loaded_config = load_yaml_configuration(str(saved_path))
        assert loaded_config == test_config


def test_extra_field_error():
    """
    Check an extra parameters in config causes error
    """

    configuration = Configuration()
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = Configuration(**configuration_dict)


def test_incorrect_interval_start_minutes(test_config_filename):
    """
    Check a history length not divisible by time resolution causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].interval_start_minutes = -1111
    with pytest.raises(
        ValueError, 
        match="interval_start_minutes must be divisible by time_resolution_minutes"
    ):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_interval_end_minutes(test_config_filename):
    """
    Check a forecast length not divisible by time resolution causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].interval_end_minutes = 1111
    with pytest.raises(
        ValueError, 
        match="interval_end_minutes must be divisible by time_resolution_minutes"
    ):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_nwp_provider(test_config_filename):
    """
    Check an unexpected nwp provider causes error
    """

    configuration = load_yaml_configuration(test_config_filename)

    configuration.input_data.nwp['ukv'].provider = "unexpected_provider"
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

    with pytest.raises(ValidationError,  match="Input should be less than or equal to 1"):
        _ = Configuration(**configuration.model_dump())

    configuration.input_data.nwp['ukv'].dropout_fraction= -0.1
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
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