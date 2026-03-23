from ocf_data_sampler.numpy_sample import convert_to_numpy_sample


def test_convert_satellite_to_numpy_sample(da_sat_like):
    t0_idx = 0
    numpy_sample = convert_to_numpy_sample({"sat": da_sat_like}, t0_idx=t0_idx)

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["satellite"] == da_sat_like.values).all()
