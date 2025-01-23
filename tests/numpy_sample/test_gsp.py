from ocf_data_sampler.load.gsp import open_gsp
import numpy as np

from ocf_data_sampler.numpy_sample import convert_gsp_to_numpy_sample, GSPSampleKey

def test_convert_gsp_to_numpy_sample(uk_gsp_zarr_path):
    
    da = (
        open_gsp(uk_gsp_zarr_path)
        .isel(time_utc=slice(0, 10))
        .sel(gsp_id=1)
    )

    numpy_sample = convert_gsp_to_numpy_sample(da)

    # Test data structure
    assert isinstance(numpy_sample, dict), "Should be dict"
    assert set(numpy_sample.keys()).issubset({
        GSPSampleKey.gsp,
        GSPSampleKey.nominal_capacity_mwp,
        GSPSampleKey.effective_capacity_mwp,
        GSPSampleKey.time_utc,
    }), "Unexpected keys"

    # Assert data content and capacity values
    assert np.array_equal(numpy_sample[GSPSampleKey.gsp], da.values), "GSP values mismatch"
    assert isinstance(numpy_sample[GSPSampleKey.time_utc], np.ndarray), "Time UTC should be numpy array"
    assert numpy_sample[GSPSampleKey.time_utc].dtype == float, "Time UTC should be float type"
    assert numpy_sample[GSPSampleKey.nominal_capacity_mwp] == da.isel(time_utc=0)["nominal_capacity_mwp"].values
    assert numpy_sample[GSPSampleKey.effective_capacity_mwp] == da.isel(time_utc=0)["effective_capacity_mwp"].values

    # Test with t0_idx
    t0_idx = 5
    numpy_sample_with_t0 = convert_gsp_to_numpy_sample(da, t0_idx=t0_idx)
    assert numpy_sample_with_t0[GSPSampleKey.t0_idx] == t0_idx, "t0_idx not correctly set"



