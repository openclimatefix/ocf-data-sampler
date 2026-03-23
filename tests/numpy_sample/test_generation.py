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
    assert "generation_time_utc" in numpy_sample

    # Assert content and capacity values
    assert np.array_equal(numpy_sample["generation"], da.sel(gen_param="generation_mw").values)
    assert isinstance(numpy_sample["generation_time_utc"], np.ndarray)
    assert numpy_sample["generation_time_utc"].dtype == float
    assert numpy_sample["capacity_mwp"] == da.sel(gen_param="capacity_mwp").isel(time_utc=0).values
