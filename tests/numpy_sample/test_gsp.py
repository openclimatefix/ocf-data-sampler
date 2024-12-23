from ocf_data_sampler.load.gsp import open_gsp

from ocf_data_sampler.numpy_sample import convert_gsp_to_numpy_sample, GSPSampleKey

def test_convert_gsp_to_numpy_sample(uk_gsp_zarr_path):

    da = (
        open_gsp(uk_gsp_zarr_path)
        .isel(time_utc=slice(0, 10))
        .sel(gsp_id=1)
    )

    # Call the function
    numpy_sample = convert_gsp_to_numpy_sample(da)

    # Assert the output type
    assert isinstance(numpy_sample, dict)

    # Assert the shape of the numpy sample
    assert (numpy_sample[GSPSampleKey.gsp] == da.values).all()

