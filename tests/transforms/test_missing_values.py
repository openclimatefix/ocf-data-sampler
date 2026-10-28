import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.transforms.missing_values import validate_and_fill_nans


@pytest.mark.parametrize(
    "values, fill_value, reject_nan, reject_inf, expected",
    [
        ([1.0, np.nan], None, False, True, [1.0, np.nan]),
        ([1.0, np.nan], 0.0, False, True, [1.0, 0.0]),
        ([np.nan, np.inf, -np.inf], -2.0, False, False, [-2.0, np.inf, -np.inf]),
        ([1.0, 2.0], None, True, True, [1.0, 2.0]),
    ],
)
def test_validate_and_fill_nans(values, fill_value, reject_nan, reject_inf, expected):
    original = np.array(values, dtype=np.float32)
    da = xr.DataArray(original)

    validate_and_fill_nans(
        da,
        fill_value,
        reject_nan=reject_nan,
        reject_inf=reject_inf,
        source="sat",
    )

    np.testing.assert_array_equal(da.data, expected)
    np.testing.assert_array_equal(original, values)
    # Guard against type promotion
    assert da.dtype == np.float32


@pytest.mark.parametrize(
    "values, fill_value, reject_nan, message",
    [
        ([1.0], 1.0, True, "source: fill_value must be None when reject_nan is True"),
        ([np.nan], None, True, "source contains NaNs"),
        ([np.nan, np.inf], 0.0, False, "source contains infinities"),
        ([-np.inf], None, False, "source contains infinities"),
    ],
)
def test_validate_and_fill_nans_rejects_disallowed_values(values, fill_value, reject_nan, message):
    da = xr.DataArray(np.array(values, dtype=np.float32))

    with pytest.raises(ValueError, match=message):
        validate_and_fill_nans(
            da,
            fill_value,
            reject_nan=reject_nan,
            reject_inf=True,
            source="source",
        )
