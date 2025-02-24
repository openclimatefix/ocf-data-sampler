import pytest
from pydantic import ValidationError

from ocf_data_sampler.config import Configuration, load_yaml_configuration


def test_default_configuration():
    """Test if the default configuration initializes without errors."""
    _ = Configuration()


def test_extra_field_error():
    """Ensure that adding an extra field to the configuration raises an error."""

    configuration = Configuration()
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    
    # Validation should fail if extra fields are present
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _ = Configuration(**configuration_dict)


def test_incorrect_interval_start_minutes(test_config_filename):
    """Check that interval_start_minutes must be divisible by time_resolution_minutes."""

    configuration = load_yaml_configuration(test_config_filename)

    # Set an invalid interval_start_minutes value
    configuration.input_data.nwp["ukv"].interval_start_minutes = -1111
    with pytest.raises(
        ValueError,
        match=r"interval_start_minutes \(-1111\) must be divisible by time_resolution_minutes \(60\)",
    ):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_interval_end_minutes(test_config_filename):
    """Ensure that interval_end_minutes must be divisible by time_resolution_minutes."""

    configuration = load_yaml_configuration(test_config_filename)

    # Set an invalid interval_end_minutes value
    configuration.input_data.nwp["ukv"].interval_end_minutes = 1111
    with pytest.raises(
        ValueError,
        match=r"interval_end_minutes \(1111\) must be divisible by time_resolution_minutes \(60\)",
    ):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_nwp_provider(test_config_filename):
    """Verify that an invalid NWP provider name triggers an error."""

    configuration = load_yaml_configuration(test_config_filename)

    # Assign an unexpected provider name
    configuration.input_data.nwp["ukv"].provider = "unexpected_provider"
    with pytest.raises(Exception, match="NWP provider"):
        _ = Configuration(**configuration.model_dump())


def test_incorrect_dropout(test_config_filename):
    """Ensure dropout timedeltas cannot be positive, but zero is allowed."""

    configuration = load_yaml_configuration(test_config_filename)

    # Check that a positive dropout timedelta is not allowed
    configuration.input_data.nwp["ukv"].dropout_timedeltas_minutes = [120]
    with pytest.raises(Exception, match="Dropout timedeltas must be negative"):
        _ = Configuration(**configuration.model_dump())

    # Check that zero is a valid dropout timedelta
    configuration.input_data.nwp["ukv"].dropout_timedeltas_minutes = [0]
    _ = Configuration(**configuration.model_dump())


def test_incorrect_dropout_fraction(test_config_filename):
    """Validate that dropout_fraction must be between 0 and 1."""

    configuration = load_yaml_configuration(test_config_filename)

    # Setting dropout_fraction to a value greater than 1 should raise an error
    configuration.input_data.nwp["ukv"].dropout_fraction = 1.1
    with pytest.raises(ValidationError, match="Input should be less than or equal to 1"):
        _ = Configuration(**configuration.model_dump())

    # Setting dropout_fraction to a negative value should also raise an error
    configuration.input_data.nwp["ukv"].dropout_fraction = -0.1
    with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
        _ = Configuration(**configuration.model_dump())


def test_inconsistent_dropout_use(test_config_filename):
    """Check that dropout_fraction and dropout_timedeltas are used correctly together."""

    configuration = load_yaml_configuration(test_config_filename)

    # Case 1: dropout_fraction is positive, but dropout_timedeltas list is empty
    configuration.input_data.satellite.dropout_fraction = 1.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = []

    with pytest.raises(
        ValueError,
        match="To dropout fraction > 0 requires a list of dropout timedeltas",
    ):
        _ = Configuration(**configuration.model_dump())

    # Case 2: dropout_timedeltas are defined, but dropout_fraction is 0
    configuration.input_data.satellite.dropout_fraction = 0.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = [-120, -60]

    with pytest.raises(
        ValueError,
        match="To use dropout timedeltas dropout fraction should be > 0",
    ):
        _ = Configuration(**configuration.model_dump())
