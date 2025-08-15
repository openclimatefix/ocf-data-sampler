from ocf_data_sampler.select.diff_channels import diff_channels
import numpy as np

def test_diff_channels(ds_nwp_ukv_time_sliced):

    # Make a copy since the function edits the inputs in-place
    da = ds_nwp_ukv_time_sliced.copy(deep=True)
    channels = [*da.channel.values]

    # This test relies on there being more than one channel in the fixture
    assert len(channels)>1

    da_diffed = diff_channels(da, accum_channels=channels[:1])

    # The diff function reduces the steps by one
    assert (da_diffed.step.values==ds_nwp_ukv_time_sliced.step.values[:-1]).all()

    # Check that these channels have not beeen changed
    expected_result = ds_nwp_ukv_time_sliced.isel(channel=slice(1, None), step=slice(None, -1))
    assert da_diffed.isel(channel=slice(1, None)).equals(expected_result)

    # Check that these channels have been properly diffed
    expected_result = (
        ds_nwp_ukv_time_sliced.diff(dim="step", label="lower")
        .isel(channel=slice(None, 1))
    ).values
    assert (da_diffed.isel(channel=slice(None, 1)).values==expected_result).all()
