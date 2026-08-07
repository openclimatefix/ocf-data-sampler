import numpy as np
import pandas as pd

from ocf_data_sampler.datasets.pvnet.sample import (
    convert_to_numpy_sample,
    make_sun_position_numpy_sample,
)
from ocf_data_sampler.load.generation import open_generation


def test_make_sun_position_numpy_sample():
    datetimes = pd.date_range("2024-06-20 12:00", "2024-06-20 16:00", freq="30min").values
    sample = make_sun_position_numpy_sample(datetimes, lon=0, lat=51.5)

    # Assertion accounting for solar coord normalisation
    assert {"solar_elevation", "solar_azimuth"} <= set(sample)
    assert np.all((sample["solar_elevation"] >= 0) & (sample["solar_elevation"] <= 1))
    assert np.all((sample["solar_azimuth"] >= 0) & (sample["solar_azimuth"] <= 1))


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    """convert_to_numpy_sample just extracts generation_mw/capacity_mwp as-is - normalising
    generation_mw to a capacity factor is normalise_dataset_dicts's job (see test_preprocess.py).
    """
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)
    numpy_sample = convert_to_numpy_sample({"generation_input": da, "generation_target": da})

    generation_mw = da.sel(gen_param="generation_mw").values
    capacity = da.sel(gen_param="capacity_mwp").values

    for key in ("generation_input", "generation_target"):
        # Assert structure
        assert isinstance(numpy_sample, dict)
        assert key in numpy_sample
        assert f"{key}_capacity_mwp" in numpy_sample
        assert f"{key}_time_utc" in numpy_sample

        # Assert content is passed through unchanged
        assert np.array_equal(numpy_sample[key], generation_mw)
        assert np.array_equal(numpy_sample[f"{key}_capacity_mwp"], capacity)
        assert isinstance(numpy_sample[f"{key}_time_utc"], np.ndarray)
        assert numpy_sample[f"{key}_time_utc"].dtype == float


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    numpy_sample = convert_to_numpy_sample({"nwp": {"ukv": ds_nwp_ukv_time_sliced}})

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["nwp_ukv"] == ds_nwp_ukv_time_sliced.values).all()


def test_convert_satellite_to_numpy_sample(da_sat_like):
    numpy_sample = convert_to_numpy_sample({"sat": da_sat_like})

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["satellite"] == da_sat_like.values).all()
