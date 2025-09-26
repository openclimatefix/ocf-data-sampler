from ocf_data_sampler.numpy_sample import NWPSampleKey, convert_nwp_to_numpy_sample


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    numpy_sample = convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced)

    # Assert output type and shape of sample
    assert isinstance(numpy_sample, dict)
    assert (numpy_sample[NWPSampleKey.nwp] == ds_nwp_ukv_time_sliced.values).all()
