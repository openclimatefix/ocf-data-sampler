from ocf_data_sampler.select.diff_channels import diff_channels


def test_diff_channels(ds_nwp_ukv_time_sliced):
    # Construct copy as function edits inputs in-place
    # Assert more than one channel in fixture
    da = ds_nwp_ukv_time_sliced.copy(deep=True)
    channels = list(da.channel.values)
    assert len(channels) > 1

    # Assert diff function reduces the steps by one
    da_diffed = diff_channels(da, accum_channels=channels[:1])
    assert (da_diffed.step.values == ds_nwp_ukv_time_sliced.step.values[:-1]).all()

    # Check these channels have not been changed
    expected_unchanged = ds_nwp_ukv_time_sliced.isel(channel=slice(1, None), step=slice(None, -1))
    assert da_diffed.isel(channel=slice(1, None)).equals(expected_unchanged)

    # Check these channels have been properly diffed
    expected_diffed = (
        ds_nwp_ukv_time_sliced.diff(dim="step", label="lower").isel(channel=slice(None, 1))
    ).values
    assert (da_diffed.isel(channel=slice(None, 1)).values == expected_diffed).all()
