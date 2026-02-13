from ocf_data_sampler.numpy_sample import convert_xarray_dict_to_numpy_sample


def test_convert_satellite_to_numpy_sample(da_sat_like):
    numpy_sample = convert_xarray_dict_to_numpy_sample({"satellite": da_sat_like})

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["satellite_actual"] == da_sat_like.values).all()
