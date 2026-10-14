import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.features.diff_channels import diff_channels


@pytest.mark.parametrize("array_factory", [xr.DataArray, LightDataArray.from_xarray])
def test_diff_channels(ds_nwp_ukv_time_sliced, array_factory):

    # Use explicit channel names for clearer testing
    channels = ["dswrf", "t"]
    da_input = ds_nwp_ukv_time_sliced.assign_coords(channel=channels)

    # Construct copy as function edits inputs in-place
    da = array_factory(da_input.copy(deep=True))

    da_diffed = diff_channels(da, accum_channels=["dswrf"])

    # Convert to xarray for easier testing
    if isinstance(da_diffed, LightDataArray):
        da_diffed = da_diffed.to_xarray()

    assert list(da_diffed["channel"].values) == ["diff_dswrf", "t"]

    # Diffing should reduce the number of steps by one
    assert (da_diffed["step"].values == da_input["step"].values[:-1]).all()

    # Undiffed channels should not have been changed
    expected_unchanged = da_input.isel(channel=slice(1, None), step=slice(None, -1))
    assert da_diffed.isel(channel=slice(1, None)).equals(expected_unchanged)

    # The diff function should agree with xarray's diff method for the diffed channels
    expected_diffed = da_input.diff(dim="step", label="lower").isel(channel=slice(None, 1))
    assert (da_diffed.isel(channel=slice(None, 1)).values == expected_diffed.values).all()
