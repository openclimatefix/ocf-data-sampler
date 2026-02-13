import numpy as np

from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.numpy_sample import convert_xarray_dict_to_numpy_sample


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)

    numpy_sample = convert_xarray_dict_to_numpy_sample({"generation": da})

    expected_keys = {
        "generation",
        "capacity_mwp",
        "time_utc",
    }

    assert isinstance(numpy_sample, dict)
    assert expected_keys.issubset(set(numpy_sample.keys()))

    assert np.array_equal(numpy_sample["generation"], da.values)
    assert isinstance(numpy_sample["time_utc"], np.ndarray)
    assert numpy_sample["time_utc"].dtype == float

    assert numpy_sample["capacity_mwp"] == da.capacity_mwp.values[0]

    # With t0_idx
    t0_idx = 5
    numpy_sample_with_t0 = convert_xarray_dict_to_numpy_sample(
        {"generation": da},
        t0_indices={"generation": t0_idx},
    )
    assert numpy_sample_with_t0["t0_idx"] == t0_idx
