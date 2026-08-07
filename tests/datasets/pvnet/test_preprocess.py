import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.datasets.pvnet.preprocess import (
    apply_dropout_to_datasets,
    fill_nans_in_dataset_dicts,
    normalise_dataset_dicts,
)


def _generation_da(generation_mw: np.ndarray, capacity_mwp: np.ndarray) -> xr.DataArray:
    """Build a generation DataArray from 2D `(time_utc, location_id)` MW and capacity arrays."""
    n_times, n_locations = generation_mw.shape
    return xr.DataArray(
        np.stack([generation_mw, capacity_mwp], axis=-1),
        coords={
            "time_utc": pd.date_range("2023-01-01", periods=n_times, freq="30min"),
            "location_id": np.arange(1, n_locations + 1),
            "gen_param": ["generation_mw", "capacity_mwp"],
        },
        dims=("time_utc", "location_id", "gen_param"),
    )


def _assert_dropout_applied(da: xr.DataArray, cutoff_time: np.datetime64):
    """Assert that all values after the cutoff time are NaN, and all values before are not NaN.
    """
    assert not np.any(np.isnan(da.sel(time_utc=slice(None, cutoff_time))))
    assert np.all(np.isnan(da.sel(time_utc=slice(cutoff_time + np.timedelta64(1, "s"), None))))


def test_fill_nans_in_dataset_dicts(config_filename):
    """Test the fill_nans_in_arrays function from configuration"""

    configuration = load_yaml_configuration(config_filename)

    # Set custom satellite and nwp values, generation is left as default 0.0
    configuration.satellite.dropout_fill_value = -1.0
    configuration.nwp["ukv"].dropout_fill_value = -2.0

    gen = np.array([1.0, np.nan, 3.0, np.nan])
    sat = np.array([1.0, np.nan, 3.0, np.nan])
    ukv = np.array([np.nan, 3.0, np.nan])

    datasets_dict = {
        "generation_input": xr.DataArray(gen),
        "generation_target": xr.DataArray(gen.copy()),
        "sat": xr.DataArray(sat),
        "nwp": {"ukv": xr.DataArray(ukv)},
    }

    datasets_dict = fill_nans_in_dataset_dicts(datasets_dict, config=configuration)

    expected_gen = np.array([1.0, 0.0, 3.0, 0.0])
    assert np.array_equal(datasets_dict["generation_input"].values, expected_gen)
    assert np.array_equal(datasets_dict["generation_target"].values, expected_gen)
    assert np.array_equal(datasets_dict["sat"].values, np.array([1.0, -1.0, 3.0, -1.0]))
    assert np.array_equal(datasets_dict["nwp"]["ukv"].values, np.array([-2.0, 3.0, -2.0]))


def test_normalise_dataset_dicts_generation():
    """Generation is normalised to a capacity factor (generation_mw / capacity_mwp)."""
    capacity_mwp = np.array([[100.0, 0.0], [100.0, 40.0]])
    generation = _generation_da(np.array([[50.0, 0.0], [100.0, 20.0]]), capacity_mwp)

    datasets_dict = {
        "generation_input": generation,
        "generation_target": generation.copy(deep=True),
    }

    datasets_dict = normalise_dataset_dicts(datasets_dict, {}, {}, {}, {})

    for key in ("generation_input", "generation_target"):
        result = datasets_dict[key]

        # generation_mw is rescaled element-wise by capacity_mwp
        normalised = result.sel(gen_param="generation_mw").values
        assert normalised[0, 0] == 0.5
        assert normalised[1, 0] == 1.0
        assert normalised[1, 1] == 0.5

        # Zero capacity normalises to 0 rather than raw MW or a division error
        assert normalised[0, 1] == 0.0

        # capacity_mwp itself is left unchanged
        assert np.array_equal(result.sel(gen_param="capacity_mwp").values, capacity_mwp)


def test_apply_dropout_to_datasets(pvnet_config_filename):
    config = load_yaml_configuration(pvnet_config_filename)

    config.generation.input.dropout_timedeltas_minutes = [-30]
    config.generation.input.dropout_fraction = 1.0

    config.satellite.dropout_timedeltas_minutes = [-60]
    config.satellite.dropout_fraction = 1.0

    generation_input = _generation_da(generation_mw=np.ones((4, 2)),  capacity_mwp=np.ones((4, 2)))
    generation_target = generation_input.copy(deep=True)

    t0 = np.datetime64("2023-01-01 00:00")

    times = pd.date_range(end=t0, periods=4, freq="30min").values

    sat = xr.DataArray(
        np.arange(4, dtype=float),
        coords={"time_utc": times},
        dims=("time_utc",),
    )

    datasets_dict = {
        "generation_input": generation_input,
        "generation_target": generation_target,
        "sat": sat
    }

    apply_dropout_to_datasets(datasets_dict, t0, config)

    # Generation dropout with a -30 minute cutoff should blank everything from t0
    _assert_dropout_applied(datasets_dict["generation_input"], t0 - np.timedelta64(30, "m"))

    # No dropout is applied to generation.target, so it should have no NaNs
    assert not np.any(np.isnan(datasets_dict["generation_target"]))

    # Satellite dropout with a -60 minute cutoff should blank everything from t0 - 60 minutes
    _assert_dropout_applied(datasets_dict["sat"], t0 - np.timedelta64(60, "m"))



