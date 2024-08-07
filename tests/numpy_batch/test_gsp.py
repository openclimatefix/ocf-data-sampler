from ocf_datapipes.batch import BatchKey
from ocf_data_sampler.load.gsp import open_gsp

from ocf_data_sampler.numpy_batch import convert_gsp_to_numpy_batch


def test_convert_gsp_to_numpy_batch(uk_gsp_zarr_path):

    da = (
        open_gsp(uk_gsp_zarr_path)
        .isel(time_utc=slice(0, 10))
        .sel(gsp_id=1)
    )

    # Call the function
    numpy_batch = convert_gsp_to_numpy_batch(da)

    # Assert the output type
    assert isinstance(numpy_batch, dict)

    # Assert the shape of the numpy batch
    assert (numpy_batch[BatchKey.gsp] == da.values).all()

