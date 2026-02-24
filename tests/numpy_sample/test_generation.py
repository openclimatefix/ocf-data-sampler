import numpy as np

from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.numpy_sample import convert_to_numpy_sample


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)
    t0_idx = 0
    numpy_sample = convert_to_numpy_sample({"generation": da}, t0_idx=t0_idx)

    # Assert structure
    assert isinstance(numpy_sample, dict)
    assert "generation" in numpy_sample
    assert "capacity_mwp" in numpy_sample
    assert "time_utc" in numpy_sample

    # Assert content and capacity values
    assert np.array_equal(numpy_sample["generation"], da.values)
    assert isinstance(numpy_sample["time_utc"], np.ndarray)
    assert numpy_sample["time_utc"].dtype == float
    assert numpy_sample["capacity_mwp"] == da.capacity_mwp.isel(time_utc=0).values

    # Assert t0_idx is passed through
    assert numpy_sample["t0_idx"] == 0
