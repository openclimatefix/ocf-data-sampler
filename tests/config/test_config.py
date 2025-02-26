import pytest
from pydantic import ValidationError
from ocf_data_sampler.config import Configuration, load_yaml_configuration

def test_default_configuration():
    """Make sure the default config initializes without any issues."""
    _ = Configuration()

def test_extra_field_error():
    """Check that extra unexpected fields throw a validation error."""

    configuration = Configuration()
    config_dict = configuration.model_dump()
    
    # Adding an unexpected field that shouldn't exist
    config_dict["extra_field"] = "unexpected_value"
    
    # Should raise an error because extra fields are not allowed
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = Configuration(**config_dict)

def test_incorrect_interval_start_minutes(test_config_filename):
    """Ensure interval_start_minutes is properly validated."""

    configuration = load_yaml_configuration(test_config_filename)

    # Assign an invalid value (should be a multiple of time_resolution_minutes)
    configuration.input_data.nwp["ukv"].interval_start_minutes = -1111

    with pytest.raises(
        ValueError,
        match=r"interval_start_minutes \(-1111\) must be divisible by time_resolution_minutes \(60\)",
    ):
        _ = Configuration(**configuration.model_dump())

def test_incorrect_interval_end_minutes(test_config_filename):
    """Make sure interval_end_minutes follows the correct divisibility rule."""

    configuration = load_yaml_configuration(test_config_filename)

    # Assigning an invalid value
    configuration.input_data.nwp["ukv"].interval_end_minutes = 1111

    with pytest.raises(
        ValueError,
        match=r"interval_end_minutes \(1111\) must be divisible by time_resolution_minutes \(60\)",
    ):
        _ = Configuration(**configuration.model_dump())

def test_incorrect_nwp_provider(test_config_filename):
    """Check if assigning an invalid provider name raises an error."""

    configuration = load_yaml_configuration(test_config_filename)

    # Setting an invalid provider name
    configuration.input_data.nwp["ukv"].provider = "invalid_provider"

    with pytest.raises(Exception, match="NWP provider"):
        _ = Configuration(**configuration.model_dump())

def test_incorrect_dropout(test_config_filename):
    """Ensure dropout time values make sense (should be negative or zero)."""

    configuration = load_yaml_configuration(test_config_filename)

    # A positive dropout timedelta should NOT be allowed
    configuration.input_data.nwp["ukv"].dropout_timedeltas_minutes = [120]

    with pytest.raises(Exception, match="Dropout timedeltas must be negative"):
        _ = Configuration(**configuration.model_dump())

    # Zero should be allowed
    configuration.input_data.nwp["ukv"].dropout_timedeltas_minutes = [0]
    _ = Configuration(**configuration.model_dump())  # This should pass

def test_incorrect_dropout_fraction(test_config_filename):
    """Make sure dropout_fraction values stay between 0 and 1."""

    configuration = load_yaml_configuration(test_config_filename)

    # Setting dropout_fraction higher than 1 should fail
    configuration.input_data.nwp["ukv"].dropout_fraction = 1.1
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        _ = Configuration(**configuration.model_dump())

    # Setting dropout_fraction lower than 0 should also fail
    configuration.input_data.nwp["ukv"].dropout_fraction = -0.1
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        _ = Configuration(**configuration.model_dump())

def test_inconsistent_dropout_use(test_config_filename):
    """Check that dropout_fraction and dropout_timedeltas are used together correctly."""

    configuration = load_yaml_configuration(test_config_filename)

    # Case 1: dropout_fraction is greater than 0, but dropout_timedeltas is empty, i.e., this should fail
    configuration.input_data.satellite.dropout_fraction = 1.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = []

    with pytest.raises(
        ValueError,
        match="To dropout fraction > 0 requires a list of dropout timedeltas",
    ):
        _ = Configuration(**configuration.model_dump())

    # Case 2: dropout_timedeltas are defined, but dropout_fraction is 0, i.e., this should fail
    configuration.input_data.satellite.dropout_fraction = 0.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = [-120, -60]

    with pytest.raises(
        ValueError,
        match="To use dropout timedeltas dropout fraction should be > 0",
    ):
        _ = Configuration(**configuration.model_dump())

    # Case 3: Both dropout_fraction is 0 and dropout_timedeltas is empty, i.e., this should pass
    configuration.input_data.satellite.dropout_timedeltas_minutes = []
    _ = Configuration(**configuration.model_dump())  # This should work fine
