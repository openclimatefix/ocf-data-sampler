from ocf_data_sampler.numpy_sample import convert_xarray_dict_to_numpy_sample


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    numpy_sample = convert_xarray_dict_to_numpy_sample(
        {"nwp": {"ukv": ds_nwp_ukv_time_sliced}}
    )

    assert isinstance(numpy_sample, dict)

    nwp_values = numpy_sample["nwp"]["ukv"]["nwp"]
    assert (nwp_values == ds_nwp_ukv_time_sliced.values).all()