import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import date_range, minutes
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.datasets.pvnet.slicing import slice_sources_by_time


def _time_series(start: np.datetime64, end: np.datetime64, freq: np.timedelta64) -> xr.DataArray:
    times = date_range(start, end, freq=freq)
    return xr.DataArray(np.ones(len(times), dtype=np.float32), coords={"time_utc": times})


def test_slice_sources_by_time_applies_dropout(config_filename):
    """Satellite and generation input are dropped out at their delays, but not the target"""
    config = load_yaml_configuration(config_filename)
    config.nwp = None
    # A probability of 1 always draws the same delay
    config.satellite.fill_value = -1.0
    config.satellite.dropout_minute_probabilities = {10: 1.0}
    config.generation.input.fill_value = -1.0
    config.generation.input.dropout_minute_probabilities = {30: 1.0}

    t0 = np.datetime64("2024-01-01T12:00")
    source_data = {
        "sat": _time_series(t0 - minutes(60), t0 + minutes(60), minutes(5)),
        "generation": _time_series(t0 - minutes(120), t0 + minutes(180), minutes(30)),
    }

    sliced = slice_sources_by_time(source_data, t0, config, "random")

    # Satellite window is [-30, 0] at 5 minutes; times after t0 - 10 minutes are dropped
    np.testing.assert_array_equal(
        sliced["sat"]["time_utc"].values,
        date_range(t0 - minutes(30), t0, freq=minutes(5)),
    )
    np.testing.assert_array_equal(np.isnan(sliced["sat"].values), [0, 0, 0, 0, 0, 1, 1])

    # Generation input window is [-60, 0] at 30 minutes; times after t0 - 30 minutes are dropped
    np.testing.assert_array_equal(np.isnan(sliced["generation_input"].values), [0, 0, 1])

    # Generation target window is [0, 120] and is never dropped out
    np.testing.assert_array_equal(
        sliced["generation_target"]["time_utc"].values,
        date_range(t0, t0 + minutes(120), freq=minutes(30)),
    )
    assert not np.isnan(sliced["generation_target"].values).any()


@pytest.mark.parametrize(
    ("sat_end_minutes", "expected_sat_nan"),
    [
        # With complete data, no dropout is used, since the probabilities sum to less than 1
        (60, [0, 0, 0, 0, 0, 0, 0]),
        # Data ends at t0 - 15 minutes, so 20 minutes is the smallest consistent delay
        (-15, [0, 0, 0, 1, 1, 1, 1]),
    ],
)
def test_slice_sources_by_time_minimal_dropout(config_filename, sat_end_minutes, expected_sat_nan):
    """Minimal mode uses the smallest configured delay consistent with the available data"""
    config = load_yaml_configuration(config_filename)
    config.nwp = None
    config.satellite.fill_value = -1.0
    config.satellite.dropout_minute_probabilities = {10: 0.2, 20: 0.2}
    config.generation.input.fill_value = -1.0
    config.generation.input.dropout_minute_probabilities = {30: 1.0}

    t0 = np.datetime64("2024-01-01T12:00")
    source_data = {
        "sat": _time_series(t0 - minutes(60), t0 + minutes(sat_end_minutes), minutes(5)),
        "generation": _time_series(t0 - minutes(120), t0 + minutes(180), minutes(30)),
    }

    sliced = slice_sources_by_time(source_data, t0, config, "minimal")

    np.testing.assert_array_equal(np.isnan(sliced["sat"].values), expected_sat_nan)

    # Generation data is complete, but its probabilities sum to 1, so the model never saw it
    # without dropout and the 30-minute delay is still applied
    np.testing.assert_array_equal(np.isnan(sliced["generation_input"].values), [0, 0, 1])
