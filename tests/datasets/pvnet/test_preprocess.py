import numpy as np
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.datasets.pvnet.preprocess import apply_dropout_to_datasets, fill_nans_in_dataset_dicts


def test_fill_nans_in_dataset_dicts(config_filename):
    """Test the fill_nans_in_arrays function from configuration"""

    configuration = load_yaml_configuration(config_filename)

    # Set custom satellite and nwp values, generation is left as default 0.0
    configuration.input_data.satellite.dropout_fill_value = -1.0
    configuration.input_data.nwp["ukv"].dropout_fill_value = -2.0

    gen = np.array([1.0, np.nan, 3.0, np.nan])
    sat = np.array([1.0, np.nan, 3.0, np.nan])
    ukv = np.array([np.nan, 3.0, np.nan])

    datasets_dict = {
        "generation": xr.DataArray(gen),
        "sat": xr.DataArray(sat),
        "nwp": {"ukv": xr.DataArray(ukv)},
    }

    datasets_dict = fill_nans_in_dataset_dicts(datasets_dict, config=configuration)

    assert np.array_equal(datasets_dict["generation"].values, np.array([1.0, 0.0, 3.0, 0.0]))
    assert np.array_equal(datasets_dict["sat"].values, np.array([1.0, -1.0, 3.0, -1.0]))
    assert np.array_equal(datasets_dict["nwp"]["ukv"].values, np.array([-2.0, 3.0, -2.0]))


def test_apply_dropout_to_datasets(pvnet_config_filename):
    config = load_yaml_configuration(pvnet_config_filename)

    # Set dropout
    config.input_data.generation.dropout_timedeltas_minutes = [-30]
    config.input_data.generation.dropout_fraction = 1.0
    config.input_data.satellite.dropout_timedeltas_minutes = []
    config.input_data.satellite.dropout_fraction = 0

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
    generation = xr.DataArray(
        np.arange(4 * 2, dtype=float).reshape(4, 2),
        coords={"time_utc": times, "location_id": [1, 2]},
        dims=("time_utc", "location_id"),
    )
    sat = xr.DataArray(
        np.arange(4, dtype=float),
        coords={"time_utc": times},
        dims=("time_utc",),
    )

    datasets_dict = {"generation": generation, "sat": sat}

    apply_dropout_to_datasets(datasets_dict, t0, config)

    ds_gen = datasets_dict["generation"]

    # Generation dropout with a -30 minute history should blank only the t0 timestep.
    assert not np.any(np.isnan(ds_gen.sel(time_utc=slice(None, "2023-01-01T11:30"))))
    assert np.all(np.isnan(ds_gen.sel(time_utc=t0)))
    assert not np.any(np.isnan(ds_gen.sel(time_utc=slice("2023-01-01T12:30", None))))

    # Satellite dropout is disabled, so the helper should leave it untouched.
    xr.testing.assert_equal(datasets_dict["sat"], sat)

