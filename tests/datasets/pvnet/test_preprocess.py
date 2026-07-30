import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.datasets.pvnet.preprocess import (
    apply_dropout_to_datasets,
    fill_nans_in_dataset_dicts,
    normalise_dataset_dicts,
)


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
    generation_mw = np.array([[50.0, 0.0], [100.0, 20.0]])
    capacity_mwp = np.array([[100.0, 0.0], [100.0, 40.0]])
    generation = xr.DataArray(
        np.stack([generation_mw, capacity_mwp], axis=-1),
        coords={
            "time_utc": ["2023-01-01T00:00", "2023-01-01T00:30"],
            "location_id": [1, 2],
            "gen_param": ["generation_mw", "capacity_mwp"],
        },
        dims=("time_utc", "location_id", "gen_param"),
    )

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


def _generation_source(generation_mw: np.ndarray, capacity_mwp: np.ndarray) -> xr.DataArray:
    """Build a generation DataArray with `n` timesteps for a single location."""
    return xr.DataArray(
        np.stack([generation_mw[:, None], capacity_mwp[:, None]], axis=-1),
        coords={
            "time_utc": pd.date_range("2023-01-01", periods=len(generation_mw), freq="30min"),
            "location_id": [1],
            "gen_param": ["generation_mw", "capacity_mwp"],
        },
        dims=("time_utc", "location_id", "gen_param"),
    )


def test_normalise_does_not_mutate_shared_source():
    """Normalising windows must not write through to the array they were sliced from.

    Time slices are views onto the eagerly-loaded source, so an in-place write would corrupt it
    for every later sample and double-normalise timesteps shared by the two windows.
    """
    src = _generation_source(np.array([100.0, 150.0, 200.0]), np.full(3, 100.0))
    before = src.values.copy()

    # Windows deliberately overlap on the middle timestep
    datasets_dict = {
        "generation_input": src.isel(time_utc=slice(0, 2)),
        "generation_target": src.isel(time_utc=slice(1, 3)),
    }
    datasets_dict = normalise_dataset_dicts(datasets_dict, {}, {}, {}, {})

    assert np.array_equal(src.values, before), "source array was mutated by normalisation"

    def generation_of(key: str) -> np.ndarray:
        return datasets_dict[key].sel(gen_param="generation_mw").values.ravel()

    assert np.array_equal(generation_of("generation_input"), [1.0, 1.5])
    # The shared timestep is normalised once, not once per window
    assert np.array_equal(generation_of("generation_target"), [1.5, 2.0])


def test_normalise_generation_is_dimension_order_agnostic():
    """`gen_param` is indexed by name, so it need not be the last dimension."""
    src = _generation_source(np.array([100.0, 150.0, 200.0]), np.full(3, 100.0))
    transposed = src.transpose("gen_param", "time_utc", "location_id")

    result = normalise_dataset_dicts({"generation_input": transposed}, {}, {}, {}, {})

    normalised = result["generation_input"].sel(gen_param="generation_mw").values.ravel()
    assert np.array_equal(normalised, [1.0, 1.5, 2.0])


def test_apply_dropout_to_datasets(pvnet_config_filename):
    config = load_yaml_configuration(pvnet_config_filename)

    # Set dropout on the input window only - generation.target has no dropout config
    config.generation.input.dropout_timedeltas_minutes = [-30]
    config.generation.input.dropout_fraction = 1.0
    config.satellite.dropout_timedeltas_minutes = []
    config.satellite.dropout_fraction = 0

    t0 = np.datetime64("2023-01-01 12:00")
    times = np.array(
        [
            "2023-01-01T11:00",
            "2023-01-01T11:30",
            "2023-01-01T12:00",
            "2023-01-01T12:30",
        ],
        dtype="datetime64[m]",
    )
    generation_mw = np.arange(4 * 2, dtype=float).reshape(4, 2)
    capacity_mwp = np.ones((4, 2))
    generation = xr.DataArray(
        np.stack([generation_mw, capacity_mwp], axis=-1),
        coords={
            "time_utc": times,
            "location_id": [1, 2],
            "gen_param": ["generation_mw", "capacity_mwp"],
        },
        dims=("time_utc", "location_id", "gen_param"),
    )
    sat = xr.DataArray(
        np.arange(4, dtype=float),
        coords={"time_utc": times},
        dims=("time_utc",),
    )

    datasets_dict = {"generation_input": generation, "sat": sat}

    apply_dropout_to_datasets(datasets_dict, t0, config)

    ds_gen = datasets_dict["generation_input"].sel(gen_param="generation_mw")
    ds_cap = datasets_dict["generation_input"].sel(gen_param="capacity_mwp")

    # Generation dropout with a -30 minute cutoff should blank everything from t0 onwards,
    # including timesteps beyond t0.
    assert not np.any(np.isnan(ds_gen.sel(time_utc=slice(None, "2023-01-01T11:30"))))
    assert np.all(np.isnan(ds_gen.sel(time_utc=slice("2023-01-01T12:00", None))))

    # capacity_mwp is dropped out along with generation_mw, using the same cutoff.
    assert not np.any(np.isnan(ds_cap.sel(time_utc=slice(None, "2023-01-01T11:30"))))
    assert np.all(np.isnan(ds_cap.sel(time_utc=slice("2023-01-01T12:00", None))))

    # Satellite dropout is disabled, so the helper should leave it untouched.
    xr.testing.assert_equal(datasets_dict["sat"], sat)

