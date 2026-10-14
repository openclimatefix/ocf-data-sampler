import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.features.diff_channels import diff_channels


@pytest.mark.parametrize("array_factory", [xr.DataArray, LightDataArray.from_xarray])
def test_diff_channels(ds_nwp_ukv_time_sliced, array_factory):

    # Construct copy as function edits inputs in-place
    # Use explicit channel names for cleaer testing
    channels = ["dswrf", "t"]
    da = array_factory(
        ds_nwp_ukv_time_sliced.copy(deep=True)
        .assign_coords(channel=channels)
    )

    # Assert diff function reduces the steps by one
    da_diffed = diff_channels(da, accum_channels=["dswrf"])
    if isinstance(da_diffed, LightDataArray):
        da_diffed = da_diffed.to_xarray()

    assert list(da_diffed["channel"].values) == ["diff_dswrf", "t"]
    assert (da_diffed["step"].values == ds_nwp_ukv_time_sliced["step"].values[:-1]).all()

    # Check these channels have not been changed
    expected_unchanged = ds_nwp_ukv_time_sliced.isel(channel=slice(1, None), step=slice(None, -1))
    assert da_diffed.isel(channel=slice(1, None)).equals(expected_unchanged)

    # Check these channels have been properly diffed
    expected_diffed = (
        ds_nwp_ukv_time_sliced.diff(dim="step", label="lower").isel(channel=slice(None, 1))
    ).values
    assert (da_diffed.isel(channel=slice(None, 1)).values == expected_diffed).all()
