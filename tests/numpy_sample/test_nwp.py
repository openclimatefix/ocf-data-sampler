from ocf_data_sampler.numpy_sample import NWPSampleKey, convert_nwp_to_numpy_sample


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    # Call the function
    numpy_sample = convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced)

    # Assert the output type
    assert isinstance(numpy_sample, dict)

    # Assert the shape of the numpy sample
    assert (numpy_sample[NWPSampleKey.nwp] == ds_nwp_ukv_time_sliced.values).all()
