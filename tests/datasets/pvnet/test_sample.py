import pandas as pd
import numpy as np

from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.datasets.pvnet.sample import (
    make_sun_position_numpy_sample, convert_to_numpy_sample
)


def test_make_sun_position_numpy_sample():
    datetimes = pd.date_range("2024-06-20 12:00", "2024-06-20 16:00", freq="30min").values
    sample = make_sun_position_numpy_sample(datetimes, lon=0, lat=51.5)

    # Assertion accounting for solar coord normalisation
    assert {"solar_elevation", "solar_azimuth"} <= set(sample)
    assert np.all((sample["solar_elevation"] >= 0) & (sample["solar_elevation"] <= 1))
    assert np.all((sample["solar_azimuth"] >= 0) & (sample["solar_azimuth"] <= 1))


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)
    t0_idx = 0
    numpy_sample = convert_to_numpy_sample({"generation": da}, t0_idx=t0_idx)

    # Assert structure
    assert isinstance(numpy_sample, dict)
    assert "generation" in numpy_sample
    assert "capacity_mwp" in numpy_sample
    assert "generation_time_utc" in numpy_sample

    # Assert content and capacity values
    assert np.array_equal(numpy_sample["generation"], da.sel(gen_param="generation_mw").values)
    assert isinstance(numpy_sample["generation_time_utc"], np.ndarray)
    assert numpy_sample["generation_time_utc"].dtype == float
    assert numpy_sample["capacity_mwp"] == da.sel(gen_param="capacity_mwp").isel(time_utc=0).values


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    t0_idx = 0
    numpy_sample = convert_to_numpy_sample(
        {"nwp": {"ukv": ds_nwp_ukv_time_sliced}},
        t0_idx=t0_idx,
    )

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["nwp_ukv"] == ds_nwp_ukv_time_sliced.values).all()


def test_convert_satellite_to_numpy_sample(da_sat_like):
    t0_idx = 0
    numpy_sample = convert_to_numpy_sample({"sat": da_sat_like}, t0_idx=t0_idx)

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["satellite"] == da_sat_like.values).all()