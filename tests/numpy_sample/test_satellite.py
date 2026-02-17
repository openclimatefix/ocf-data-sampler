import pandas as pd

from ocf_data_sampler.numpy_sample import convert_to_numpy_sample


def test_convert_satellite_to_numpy_sample(da_sat_like):
    t0 = pd.Timestamp(da_sat_like.time_utc.values[0])
    numpy_sample = convert_to_numpy_sample({"sat": da_sat_like}, t0=t0)

    assert isinstance(numpy_sample, dict)
    assert (numpy_sample["satellite_actual"] == da_sat_like.values).all()
