import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.config.model import Channel, NWPChannel
from ocf_data_sampler.datasets.pvnet.preprocess import (
    apply_source_dropout,
    build_normalisation_arrays,
    normalise_source_data,
    preprocess_source_data,
    validate_and_fill_source_nans,
)


def _generation_da(generation_mw: np.ndarray, capacity_mwp: np.ndarray) -> xr.DataArray:
    """Build a generation DataArray from 2D `(time_utc, location_id)` MW and capacity arrays."""
    n_times, n_locations = generation_mw.shape
    return xr.DataArray(
        np.stack([generation_mw, capacity_mwp], axis=-1),
        coords={
            "time_utc": pd.date_range("2023-01-01", periods=n_times, freq="30min"),
            "location_id": [f"loc_{i}" for i in range(1, n_locations + 1)],
            "gen_param": ["generation_mw", "capacity_mwp"],
        },
        dims=("time_utc", "location_id", "gen_param"),
    )


def _assert_dropout_applied(da: xr.DataArray, cutoff_time: np.datetime64):
    """Check that the cutoff is retained and later observations are NaN."""
    assert not np.any(np.isnan(da.sel(time_utc=slice(None, cutoff_time))))
    assert np.all(np.isnan(da.sel(time_utc=slice(cutoff_time + np.timedelta64(1, "s"), None))))


@pytest.mark.parametrize(
    "fill_value, expected",
    [
        (None, [1.0, np.nan, 3.0, np.nan]),
        (0.0, [1.0, 0.0, 3.0, 0.0]),
        (-2.0, [1.0, -2.0, 3.0, -2.0]),
    ],
)
def test_validate_and_fill_source_nans(config_filename, fill_value, expected):
    """Fill generation and NWP inputs only when configured, including an explicit zero."""

    configuration = load_yaml_configuration(config_filename)
    configuration.generation.target = None

    # Zero is a valid fill value; None preserves missing observations.
    configuration.generation.input.fill_value = fill_value
    configuration.nwp["ukv"].fill_value = fill_value
    configuration.satellite.fill_value = -1.0

    gen = np.array([1.0, np.nan, 3.0, np.nan])
    sat = np.array([1.0, np.nan, 3.0, np.nan])

    source_data = {
        "generation_input": xr.DataArray(gen),
        "sat": xr.DataArray(sat),
        "nwp": {"ukv": xr.DataArray(gen.copy())},
    }

    validate_and_fill_source_nans(source_data, config=configuration)

    np.testing.assert_array_equal(source_data["generation_input"].values, expected)
    np.testing.assert_array_equal(source_data["nwp"]["ukv"].values, expected)
    assert np.array_equal(source_data["sat"].values, np.array([1.0, -1.0, 3.0, -1.0]))


def test_apply_source_dropout(config_filename):
    config = load_yaml_configuration(config_filename)
    config.nwp = None

    config.generation.input.dropout_minute_probabilities = {30: 1.0}
    config.generation.input.fill_value = -1.0
    config.satellite.dropout_minute_probabilities = {60: 1.0}
    config.satellite.fill_value = -2.0

    generation_input = _generation_da(generation_mw=np.ones((4, 2)),  capacity_mwp=np.ones((4, 2)))
    generation_target = generation_input.copy(deep=True)

    t0 = np.datetime64("2023-01-01 00:00")

    times = pd.date_range(end=t0, periods=4, freq="30min").values

    sat = xr.DataArray(
        np.arange(4, dtype=float),
        coords={"time_utc": times},
        dims=("time_utc",),
    )

    source_data = {
        "generation_input": generation_input,
        "generation_target": generation_target,
        "sat": sat
    }

    apply_source_dropout(source_data, t0, config)

    # Generation dropout with a 30-minute delay should blank everything from t0
    _assert_dropout_applied(
        source_data["generation_input"], t0 - np.timedelta64(30, "m"),
    )

    # No dropout is applied to generation.target, so it should have no NaNs
    assert not np.any(np.isnan(source_data["generation_target"]))

    # Satellite dropout with a 60-minute delay should blank everything from t0 - 60 minutes
    _assert_dropout_applied(source_data["sat"], t0 - np.timedelta64(60, "m"))

    validate_and_fill_source_nans(source_data, config)
    assert (source_data["generation_input"].sel(time_utc=slice(t0, None)) == -1.0).all()
    assert (
        source_data["sat"].sel(time_utc=slice(t0 - np.timedelta64(30, "m"), None)) == -2.0
    ).all()
    xr.testing.assert_equal(source_data["generation_target"], generation_target)


def test_normalise_channels(config_filename):
    config = load_yaml_configuration(config_filename)
    config.satellite.channels = [
        Channel(name="b", mean=2, std=2, clip_min=0, clip_max=4),
        Channel(name="a", mean=10, std=2),
    ]
    config.nwp["ukv"].channels = [
        NWPChannel(name="b", accumulated=True, mean=2, std=2, clip_min=0, clip_max=4),
        NWPChannel(name="a", mean=10, std=2),
    ]
    sat = xr.DataArray(np.array([[-2, 2, 8], [6, 10, np.nan]], dtype=np.float32)[None, :, None, :])
    nwp = sat.copy(deep=True)
    datasets = {"sat": sat, "nwp": {"ukv": nwp}}

    normalise_source_data(datasets, build_normalisation_arrays(config))

    expected = np.array([[-1, 0, 1], [-2, 0, np.nan]])[None, :, None, :]
    for da in (sat, nwp):
        assert da.dtype == np.float32
        np.testing.assert_allclose(da.values, expected)


def test_preprocess_source_data_preserves_original_arrays(config_filename):
    """Preprocessing updates the supplied dictionary but preserves its original NumPy data."""
    config = load_yaml_configuration(config_filename)
    config.generation.input.dropout_minute_probabilities = {30: 1.0}
    config.generation.input.fill_value = -4.0
    config.satellite.channels = [Channel(name="a", mean=2, std=2, clip_min=0, clip_max=6)]
    config.satellite.dropout_minute_probabilities = {30: 1.0}
    config.satellite.fill_value = -3.0
    config.nwp["ukv"].channels = [
        NWPChannel(name="a", accumulated=True, mean=2, std=2, clip_min=0, clip_max=6),
    ]

    generation = _generation_da(
        np.array([[5], [8], [2]], dtype=np.float32),
        np.full((3, 1), 10, dtype=np.float32),
    )
    times = generation.time_utc.values
    sat = xr.DataArray(
        np.array([-2, np.nan, 8], dtype=np.float32)[:, None, None, None],
        coords={"time_utc": times, "channel": ["a"], "x": [0], "y": [0]},
        dims=("time_utc", "channel", "x", "y"),
    )
    nwp = xr.DataArray(
        np.array([2, 6, 14], dtype=np.float32)[:, None, None, None],
        coords={"step": times - times[0], "channel": ["a"], "x": [0], "y": [0]},
        dims=("step", "channel", "x", "y"),
    )
    source_data = {
        "generation_input": generation,
        "generation_target": generation.copy(deep=True),
        "sat": sat,
        "nwp": {"ukv": nwp},
    }
    # Keep references to the actual input arrays, since preprocessing replaces .data.
    original_array_references = [
        source_data["generation_input"].data,
        source_data["generation_target"].data,
        source_data["sat"].data,
        source_data["nwp"]["ukv"].data,
    ]
    # Make copies so we can detect if the original arrays are modified
    original_array_copies = [array.copy() for array in original_array_references]

    preprocess_source_data(source_data, times[-1], config, build_normalisation_arrays(config))

    processed_arrays = [
        source_data["generation_input"].data,
        source_data["generation_target"].data,
        source_data["sat"].data,
        source_data["nwp"]["ukv"].data,
    ]
    for original_reference, original_copy, processed in zip(
        original_array_references, original_array_copies, processed_arrays, strict=True,
    ):
        # Check that the processing actually returned different arrays from the original, else the
        # following check that the original NumPy arrays are unmodified becomes a dummy
        assert not np.array_equal(processed, original_copy, equal_nan=True)

        # Check that the original arrays are unmodified
        np.testing.assert_array_equal(original_reference, original_copy)

