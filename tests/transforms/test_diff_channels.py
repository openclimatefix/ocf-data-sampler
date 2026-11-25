import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.transforms.diff_channels import diff_channels_along_step


@pytest.fixture()
def da_nwp():
    steps = np.arange(3) * np.timedelta64(1, "h")
    channels = ["dswrf", "t"]
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    shape = (len(steps), len(channels), len(x), len(y))
    return xr.DataArray(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        coords={
            "step": steps.astype("timedelta64[ns]"),
            "channel": channels,
            "x_osgb": x,
            "y_osgb": y,
        },
        dims=("step", "channel", "x_osgb", "y_osgb"),
    )


@pytest.mark.parametrize("array_factory", [xr.DataArray, LightDataArray.from_xarray])
def test_diff_channels_along_step(da_nwp, array_factory):
    da_input = da_nwp
    da = array_factory(da_input.copy(deep=True))

    da_diffed = diff_channels_along_step(da, channels=["dswrf"])

    # Convert to xarray for easier testing
    if isinstance(da_diffed, LightDataArray):
        da_diffed = da_diffed.to_xarray()
        da = da.to_xarray()

    # Differencing should leave the input data and coordinates unchanged
    xr.testing.assert_identical(da, da_input)

    assert list(da_diffed["channel"].values) == ["diff_dswrf", "t"]

    # Diffing should reduce the number of steps by one
    assert (da_diffed["step"].values == da_input["step"].values[:-1]).all()

    # Undiffed channels should not have been changed
    expected_unchanged = da_input.sel(channel="t").isel(step=slice(None, -1))
    assert da_diffed.sel(channel="t").equals(expected_unchanged)

    # The diff function should agree with xarray's diff method for the diffed channels
    expected_diffed = da_input.sel(channel="dswrf").diff(dim="step", label="lower")
    assert (da_diffed.sel(channel="diff_dswrf").values == expected_diffed.values).all()


def test_diff_channels_along_step_missing_channel(da_nwp):
    with pytest.raises(
        ValueError,
        match=r"channels=\['dswrf', 'missing'\] not all available in "
        r"all_channels=\['dswrf', 't'\]",
    ):
        diff_channels_along_step(da_nwp, channels=["dswrf", "missing"])
