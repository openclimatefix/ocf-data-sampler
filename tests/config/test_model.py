from operator import attrgetter

import pytest
from pydantic import ValidationError

from ocf_data_sampler.config.model import DropoutMixin, NaNPolicyMixin, PVNetDataConfig

_MINIMAL_SAMPLING_GRID = {"locations_csv_path": "locations.csv", "t0_resolution_minutes": 30}


def _first_nwp_provider(config):
    return next(iter(config.nwp.keys()))


def _validate_configuration(config):
    """Recreate config instance from dict to trigger validation."""
    return PVNetDataConfig(**config.model_dump())


def test_default_configuration():
    """Test default pydantic class - sampling_grid is the only required field"""
    _ = PVNetDataConfig(sampling_grid=_MINIMAL_SAMPLING_GRID)


@pytest.mark.parametrize("resolution", [30, 60])
def test_sampling_grid_accepts_observation_resolution_multiples(base_config, resolution):
    configuration = base_config

    configuration.sampling_grid.t0_resolution_minutes = resolution
    _validate_configuration(configuration)


def test_sampling_grid_does_not_require_nwp_resolution_multiple(base_config):
    configuration = base_config
    configuration.satellite = None
    configuration.generation = None
    # A 30-minute sampling cadence need not be a multiple of the 60-minute NWP resolution.
    configuration.sampling_grid.t0_resolution_minutes = 30

    _validate_configuration(configuration)


def test_extra_field_error():
    """
    Check an extra parameters in config causes error
    """
    configuration = PVNetDataConfig(sampling_grid=_MINIMAL_SAMPLING_GRID)
    configuration_dict = configuration.model_dump()
    configuration_dict["extra_field"] = "extra_value"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PVNetDataConfig(**configuration_dict)


def test_incorrect_interval_start_minutes(base_config):
    """
    Check a history length not divisible by time resolution causes error
    """
    configuration = base_config
    provider = _first_nwp_provider(configuration)
    configuration.nwp[provider].interval_start_minutes = -1111
    with pytest.raises(
        ValueError,
        match=r"interval_start_minutes \(-1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _validate_configuration(configuration)


def test_incorrect_interval_end_minutes(base_config):
    """
    Check a forecast length not divisible by time resolution causes error
    """
    configuration = base_config
    provider = _first_nwp_provider(configuration)
    configuration.nwp[provider].interval_end_minutes = 1111
    with pytest.raises(
        ValueError,
        match=r"interval_end_minutes \(1111\) "
        r"must be divisible by time_resolution_minutes \(60\)",
    ):
        _validate_configuration(configuration)


def test_incorrect_nwp_provider(base_config):
    """
    Check an unexpected nwp provider causes error
    """
    configuration = base_config
    provider = _first_nwp_provider(configuration)
    configuration.nwp[provider].provider = "unexpected_provider"
    with pytest.raises(ValidationError, match="Unknown NWP provider"):
        _validate_configuration(configuration)


def test_dropout_probability_sum():
    """Individually valid probabilities must also form a valid distribution."""
    with pytest.raises(ValidationError, match="Probabilities must sum to at most 1"):
        DropoutMixin(dropout_minute_probabilities={60: 0.8, 120: 0.4})


def test_fill_value_requires_reject_nan_false():
    # NaNs are rejected by default, so filling must opt out explicitly.
    with pytest.raises(ValidationError, match="fill_value requires reject_nan to be false"):
        NaNPolicyMixin(fill_value=0.0)


def test_dropout_requires_fill_value():
    with pytest.raises(ValidationError, match="fill_value must be set"):
        DropoutMixin(dropout_minute_probabilities={60: 0.2}, reject_nan=False)

    # Zero is a valid replacement value, rather than an absent setting.
    config = DropoutMixin(dropout_minute_probabilities={60: 0.2}, fill_value=0.0, reject_nan=False)
    assert config.fill_value == 0.0


@pytest.mark.parametrize("window_path", ["satellite", "generation.input"])
@pytest.mark.parametrize("delay_minutes", [30, 60])
def test_dropout_rejects_cutoffs_at_or_after_interval_end(base_config, window_path, delay_minutes):
    config = base_config
    window = attrgetter(window_path)(config)
    window.interval_start_minutes = -120
    window.interval_end_minutes = -60
    window.dropout_minute_probabilities = {delay_minutes: 0.5, 90: 0.5}
    window.fill_value = 0.0
    window.reject_nan = False

    with pytest.raises(ValidationError, match="must place the cutoff before the interval end"):
        _validate_configuration(config)

    window.dropout_minute_probabilities = {90: 1.0}
    _validate_configuration(config)


@pytest.mark.parametrize("source_name", ["satellite", "ukv"])
def test_duplicate_channel_names(base_config, source_name):
    config = base_config
    sources = {"satellite": config.satellite, "ukv": config.nwp["ukv"]}
    source = sources[source_name]
    source.channels.append(source.channels[0].model_copy())

    with pytest.raises(ValidationError, match="Channel names must be unique"):
        _validate_configuration(config)


def test_generation_interval_divisibility_raises(base_config):
    """generation.input/target must be divisible by generation.time_resolution_minutes."""
    configuration = base_config
    configuration.generation.input.interval_start_minutes = -45

    with pytest.raises(
        ValueError,
        match=r"generation\.input\.interval_start_minutes \(-45\) must be divisible by "
        r"generation\.time_resolution_minutes \(30\)",
    ):
        _validate_configuration(configuration)


def test_generation_requires_input_or_target(base_config):
    """At least one of generation.input or generation.target must be configured."""
    configuration = base_config
    configuration.generation.input = None
    configuration.generation.target = None

    with pytest.raises(
        ValueError,
        match=r"At least one of `generation\.input` or `generation\.target` must be configured",
    ):
        _validate_configuration(configuration)
