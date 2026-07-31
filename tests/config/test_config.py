import pytest
from pydantic import ValidationError

from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig

_MINIMAL_SAMPLING_GRID = {"locations_csv_path": "locations.csv", "t0_resolution_minutes": 30}


def _load_config_and_provider(config_path):
    config = load_yaml_configuration(config_path)
    provider = next(iter(config.nwp.root.keys()))
    return config, provider


def _validate_configuration(config):
    """Recreate config instance from dict to trigger validation."""
    return PVNetDataConfig(**config.model_dump())


def test_default_configuration():
    """Test default pydantic class - sampling_grid is the only required field"""
    _ = PVNetDataConfig(sampling_grid=_MINIMAL_SAMPLING_GRID)


def test_extra_field_error():
    """
    Check an extra parameters in config causes error
    """
    configuration = PVNetDataConfig(sampling_grid=_MINIMAL_SAMPLING_GRID)
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PVNetDataConfig(**configuration_dict)


def test_incorrect_interval_start_minutes(config_filename):
    """
    Check a history length not divisible by time resolution causes error
    """
    configuration, provider = _load_config_and_provider(config_filename)
    configuration.nwp[provider].interval_start_minutes = -1111
    with pytest.raises(
        ValueError,
        match=r"interval_start_minutes \(-1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _validate_configuration(configuration)


def test_incorrect_interval_end_minutes(config_filename):
    """
    Check a forecast length not divisible by time resolution causes error
    """
    configuration, provider = _load_config_and_provider(config_filename)
    configuration.nwp[provider].interval_end_minutes = 1111
    with pytest.raises(
        ValueError,
        match=r"interval_end_minutes \(1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _validate_configuration(configuration)


def test_incorrect_nwp_provider(config_filename):
    """
    Check an unexpected nwp provider causes error
    """
    configuration, provider = _load_config_and_provider(config_filename)
    configuration.nwp[provider].provider = "unexpected_provider"
    with pytest.raises(ValidationError, match="Unknown NWP provider"):
        _validate_configuration(configuration)


def test_nwp_provider_is_canonicalized(config_filename):
    """NWP provider names are stored using their canonical lowercase spelling."""
    configuration, provider = _load_config_and_provider(config_filename)
    configuration.nwp[provider].provider = "UKV"

    validated = _validate_configuration(configuration)

    assert validated.nwp[provider].provider == "ukv"


def test_incorrect_dropout(config_filename):
    """
    Check a dropout timedelta over 0 causes error and 0 doesn't
    """
    configuration, provider = _load_config_and_provider(config_filename)

    # Check that a positive number is not allowed
    configuration.nwp[provider].dropout_timedeltas_minutes = [120]
    with pytest.raises(Exception, match="Dropout timedeltas must be negative"):
        _validate_configuration(configuration)

    # Check that zero is allowed
    configuration.nwp[provider].dropout_timedeltas_minutes = [0]
    _validate_configuration(configuration)


def test_incorrect_dropout_fraction(config_filename):
    """
    Check dropout fraction outside of range causes error
    """
    configuration, provider = _load_config_and_provider(config_filename)

    configuration.nwp[provider].dropout_fraction = 1.1
    with pytest.raises(ValidationError, match=r"Dropout fractions must be in range *"):
        _validate_configuration(configuration)

    configuration.nwp[provider].dropout_fraction = -0.1
    with pytest.raises(ValidationError, match=r"Dropout fractions must be in range *"):
        _validate_configuration(configuration)

    configuration.nwp[provider].dropout_fraction = [1.0, 0.1]
    with pytest.raises(ValidationError, match=r"The sum of dropout fractions must be in range *"):
        _validate_configuration(configuration)

    configuration.nwp[provider].dropout_fraction = [-0.1, 1.1]
    with pytest.raises(ValidationError, match=r"All dropout fractions must be in range *"):
        _validate_configuration(configuration)

    configuration.nwp[provider].dropout_fraction = []
    with pytest.raises(ValidationError, match="List cannot be empty"):
        _validate_configuration(configuration)


def test_dropout_fraction_list_length_matches_timedeltas(config_filename):
    """List dropout fractions must align with dropout timedeltas one-to-one."""
    configuration, provider = _load_config_and_provider(config_filename)

    configuration.nwp[provider].dropout_timedeltas_minutes = [-60, -120]
    configuration.nwp[provider].dropout_fraction = [0.5]

    with pytest.raises(
        ValidationError,
        match="must have the same length as `dropout_timedeltas_minutes`",
    ):
        _validate_configuration(configuration)


def test_inconsistent_dropout_use(config_filename):
    """
    Check dropout fraction outside of range causes error
    """
    configuration = load_yaml_configuration(config_filename)
    configuration.satellite.dropout_fraction = 1.0
    configuration.satellite.dropout_timedeltas_minutes = []
    with pytest.raises(
        ValueError,
        match="To dropout fraction > 0 requires a list of dropout timedeltas",
    ):
        _validate_configuration(configuration)

    configuration.satellite.dropout_fraction = 0.0
    configuration.satellite.dropout_timedeltas_minutes = [-120, -60]
    with pytest.raises(
        ValueError,
        match="To use dropout timedeltas dropout fraction should be > 0",
    ):
        _validate_configuration(configuration)


def test_accum_channels_validation(config_filename):
    """Test accum_channels validation with required normalization constants."""
    config, nwp_name = _load_config_and_provider(config_filename)

    # Test invalid channel scenario
    invalid_config = config.model_copy(deep=True)
    invalid_nwp = invalid_config.nwp.root[nwp_name]
    invalid_nwp.accum_channels = ["invalid_channel"]

    # Verify exact error message
    expected_error = (
        rf"nwp.{nwp_name}\n"
        fr"  Value error, NWP provider '{nwp_name}': all values in 'accum_channels' "
        r"should be present in 'channels'\. "
        r"Extra values found: {'invalid_channel'}.*"
    )
    with pytest.raises(ValidationError, match=expected_error):
        _validate_configuration(invalid_config)


def test_generation_interval_divisibility_raises(config_filename):
    """generation.input/target must be divisible by generation.time_resolution_minutes."""
    configuration = load_yaml_configuration(config_filename)
    configuration.generation.input.interval_start_minutes = -45

    with pytest.raises(
        ValueError,
        match=r"generation\.input\.interval_start_minutes \(-45\) must be divisible by "
        r"generation\.time_resolution_minutes \(30\)",
    ):
        _validate_configuration(configuration)


def test_generation_requires_input_or_target(config_filename):
    """At least one of generation.input or generation.target must be configured."""
    configuration = load_yaml_configuration(config_filename)
    configuration.generation.input = None
    configuration.generation.target = None

    with pytest.raises(
        ValueError,
        match=r"At least one of `generation\.input` or `generation\.target` must be configured",
    ):
        _validate_configuration(configuration)
