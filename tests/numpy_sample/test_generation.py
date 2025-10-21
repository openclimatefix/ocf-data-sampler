import numpy as np

from ocf_data_sampler.load.generation import open_generation
from ocf_data_sampler.numpy_sample import GenerationSampleKey, convert_generation_to_numpy_sample


def test_convert_generation_to_numpy_sample(generation_zarr_path):
    da = open_generation(generation_zarr_path).isel(time_utc=slice(0, 10)).sel(location_id=1)
    numpy_sample = convert_generation_to_numpy_sample(da)

    # Assert structure
    expected_keys = {
        GenerationSampleKey.generation,
        GenerationSampleKey.effective_capacity_mwp,
        GenerationSampleKey.time_utc,
    }
    assert isinstance(numpy_sample, dict)
    assert set(numpy_sample) <= expected_keys

    # Assert content and capacity values
    assert np.array_equal(numpy_sample[GenerationSampleKey.generation], da.values)
    assert isinstance(numpy_sample[GenerationSampleKey.time_utc], np.ndarray)
    assert numpy_sample[GenerationSampleKey.time_utc].dtype == float

    assert numpy_sample[GenerationSampleKey.effective_capacity_mwp] == (
        da.effective_capacity_mwp.isel(time_utc=0).values
    )

    # With t0_idx
    t0_idx = 5
    numpy_sample_with_t0 = convert_generation_to_numpy_sample(da, t0_idx=t0_idx)
    assert numpy_sample_with_t0[GenerationSampleKey.t0_idx] == t0_idx
