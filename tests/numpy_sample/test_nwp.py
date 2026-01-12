from ocf_data_sampler.numpy_sample import NWPSampleKey, convert_nwp_to_numpy_sample


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    numpy_sample = convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced)
    assert isinstance(numpy_sample, dict)

    # Assert output type and shape of sample
    nwp_values = numpy_sample["nwp"]["nwp"]["nwp"]
    assert (nwp_values == ds_nwp_ukv_time_sliced.values).all()
