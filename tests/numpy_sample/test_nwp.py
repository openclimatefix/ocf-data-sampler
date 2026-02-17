import pandas as pd

from ocf_data_sampler.numpy_sample import convert_to_numpy_sample


def test_convert_nwp_to_numpy_sample(ds_nwp_ukv_time_sliced):
    init_time = pd.Timestamp(ds_nwp_ukv_time_sliced.init_time_utc.values[0])
    t0 = init_time + pd.Timedelta(ds_nwp_ukv_time_sliced.step.values[0])
    numpy_sample = convert_to_numpy_sample({"nwp": {"ukv": ds_nwp_ukv_time_sliced}}, t0=t0)

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["nwp"]["ukv"]["nwp"] == ds_nwp_ukv_time_sliced.values).all()
