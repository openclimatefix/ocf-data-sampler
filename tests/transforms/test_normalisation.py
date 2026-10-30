import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.lightarray import LightDataArray
from ocf_data_sampler.transforms.normalisation import (
    clip_and_standardise,
    normalise_generation_by_capacity,
)


@pytest.mark.parametrize("array_factory", [xr.DataArray, LightDataArray.from_xarray])
def test_clip_and_standardise(array_factory):
    data = np.array([[-2, 6], [8, 14]], dtype=np.float32)
    original = data.copy()
    da = array_factory(xr.DataArray(data))

    clip_and_standardise(
        da,
        clip_min=np.array([0, 8], dtype=np.float32),
        clip_max=np.array([4, 12], dtype=np.float32),
        mean=np.array([2, 10], dtype=np.float32),
        std=np.array([2, 4], dtype=np.float32),
    )

    # -2 is clipped up to 0, then standardised to (0 - 2) / 2 = -1
    # 6 is clipped up to 8, then standardised to (8 - 10) / 4 = -0.5
    # 8 is clipped down to 4, then standardised to (4 - 2) / 2 = 1
    # 14 is clipped down to 12, then standardised to (12 - 10) / 4 = 0.5
    np.testing.assert_allclose(da.data, [[-1, -0.5], [1, 0.5]])
    assert da.data.dtype == np.float32
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("array_factory", [xr.DataArray, LightDataArray.from_xarray])
def test_normalise_generation_by_capacity(array_factory):
    """Generation is normalised to a capacity factor (generation_mw / capacity_mwp)."""
    capacity_mwp = np.array([[100.0, 0.0], [100.0, 40.0]], dtype=np.float32)
    generation_mw = np.array([[50.0, 0.0], [100.0, 20.0]], dtype=np.float32)
    data = np.stack([generation_mw, capacity_mwp], axis=-1)
    original = data.copy()
    da = array_factory(
        xr.DataArray(
            data,
            coords={"gen_param": ["generation_mw", "capacity_mwp"]},
            dims=("time_utc", "location_id", "gen_param"),
        )
    )

    normalise_generation_by_capacity(da)

    # Zero capacity normalises to 0 rather than a division error.
    np.testing.assert_array_equal(da.isel(gen_param=0).values, [[0.5, 0.0], [1.0, 0.5]])
    np.testing.assert_array_equal(da.isel(gen_param=1).values, capacity_mwp)
    assert da.data.dtype == np.float32
    np.testing.assert_array_equal(data, original)
