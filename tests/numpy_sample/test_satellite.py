
from ocf_data_sampler.numpy_sample import SatelliteSampleKey, convert_satellite_to_numpy_sample


def test_convert_satellite_to_numpy_sample(da_sat_like):
    numpy_sample = convert_satellite_to_numpy_sample(da_sat_like)

    # Assert output type and shape of sample
    assert isinstance(numpy_sample, dict)
    assert (numpy_sample[SatelliteSampleKey.satellite_actual] == da_sat_like.values).all()
