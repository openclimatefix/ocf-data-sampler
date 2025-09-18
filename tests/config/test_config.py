import pytest
from pydantic import ValidationError
from ocf_data_sampler.config import Configuration, load_yaml_configuration


def _load_config_and_provider(config_path):
    cfg = load_yaml_configuration(config_path)
    provider = next(iter(cfg.input_data.nwp.root.keys()))
    return cfg, provider


def _revalidate(cfg):
    return Configuration(**cfg.model_dump())


def test_default_configuration(test_config_gsp_path):
    """Test default pydantic class"""
    load_yaml_configuration(test_config_gsp_path)


def test_extra_field_error(test_config_gsp_path):
    """
    Check an extra parameters in config causes error
    """
    configuration = load_yaml_configuration(test_config_gsp_path)
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Configuration(**configuration_dict)


def test_incorrect_interval_start_minutes(test_config_filename):
    """
    Check a history length not divisible by time resolution causes error
    """
    configuration, provider = _load_config_and_provider(test_config_filename)
    configuration.input_data.nwp[provider].interval_start_minutes = -1111
    with pytest.raises(
        ValueError,
        match=r"interval_start_minutes \(-1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _revalidate(configuration)


def test_incorrect_interval_end_minutes(test_config_filename):
    """
    Check a forecast length not divisible by time resolution causes error
    """
    configuration, provider = _load_config_and_provider(test_config_filename)
    configuration.input_data.nwp[provider].interval_end_minutes = 1111
    with pytest.raises(
        ValueError,
        match=r"interval_end_minutes \(1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _revalidate(configuration)


def test_incorrect_nwp_provider(test_config_filename):
    """
    Check an unexpected nwp provider causes error
    """
    configuration, provider = _load_config_and_provider(test_config_filename)
    configuration.input_data.nwp[provider].provider = "unexpected_provider"
    with pytest.raises(Exception, match="NWP provider"):
        _revalidate(configuration)


def test_incorrect_dropout(test_config_filename):
    """
    Check a dropout timedelta over 0 causes error and 0 doesn't
    """
    configuration, provider = _load_config_and_provider(test_config_filename)

    # Check that a positive number is not allowed
    configuration.input_data.nwp[provider].dropout_timedeltas_minutes = [120]
    with pytest.raises(Exception, match="Dropout timedeltas must be negative"):
        _revalidate(configuration)

    # Check that zero is allowed
    configuration.input_data.nwp[provider].dropout_timedeltas_minutes = [0]
    _revalidate(configuration)


def test_incorrect_dropout_fraction(test_config_filename):
    """
    Check dropout fraction outside of range causes error
    """
    configuration, provider = _load_config_and_provider(test_config_filename)

    configuration.input_data.nwp[provider].dropout_fraction = 1.1
    with pytest.raises(ValidationError, match=r"Dropout fractions must be in range *"):
        _revalidate(configuration)

    configuration.input_data.nwp[provider].dropout_fraction = -0.1
    with pytest.raises(ValidationError, match=r"Dropout fractions must be in range *"):
        _revalidate(configuration)

    configuration.input_data.nwp[provider].dropout_fraction = [1.0, 0.1]
    with pytest.raises(ValidationError, match=r"The sum of dropout fractions must be in range *"):
        _revalidate(configuration)

    configuration.input_data.nwp[provider].dropout_fraction = [-0.1, 1.1]
    with pytest.raises(ValidationError, match=r"All dropout fractions must be in range *"):
        _revalidate(configuration)

    configuration.input_data.nwp[provider].dropout_fraction = []
    with pytest.raises(ValidationError, match="List cannot be empty"):
        _revalidate(configuration)


def test_inconsistent_dropout_use(test_config_filename):
    """
    Check dropout fraction outside of range causes error
    """
    configuration = load_yaml_configuration(test_config_filename)
    configuration.input_data.satellite.dropout_fraction = 1.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = []
    with pytest.raises(
        ValueError,
        match="To dropout fraction > 0 requires a list of dropout timedeltas",
    ):
        _revalidate(configuration)

    configuration.input_data.satellite.dropout_fraction = 0.0
    configuration.input_data.satellite.dropout_timedeltas_minutes = [-120, -60]
    with pytest.raises(
        ValueError,
        match="To use dropout timedeltas dropout fraction should be > 0",
    ):
        _revalidate(configuration)


def test_accum_channels_validation(test_config_filename):
    """Test accum_channels validation with required normalization constants."""
    config = load_yaml_configuration(test_config_filename)
    nwp_name, _ = next(iter(config.input_data.nwp.root.items()))

    # Test invalid channel scenario
    invalid_config = config.model_copy(deep=True)
    invalid_nwp = invalid_config.input_data.nwp.root[nwp_name]
    invalid_nwp.accum_channels = ["invalid_channel"]

    # Verify exact error message
    expected_error = (
        rf"input_data.nwp.{nwp_name}\n"
        fr"  Value error, NWP provider '{nwp_name}': all values in 'accum_channels' "
        r"should be present in 'channels'\. "
        r"Extra values found: {'invalid_channel'}.*"
    )
    with pytest.raises(ValidationError, match=expected_error):
        _revalidate(invalid_config)


def test_configuration_requires_site_or_gsp():
    """
    Test that Configuration raises an error if both site and gsp are None in input_data.
    """
    with pytest.raises(ValidationError, match="You must provide either `site` or `gsp`"):
        Configuration()
