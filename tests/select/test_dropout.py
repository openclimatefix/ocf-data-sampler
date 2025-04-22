import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocf_data_sampler.select.dropout import simulate_dropout


@pytest.fixture
def t0():
    """Reference time for test cases"""
    return pd.Timestamp("2024-01-01 04:00:00")


@pytest.fixture
def da_sample(t0):
    """Create t0-centered test data with 5-minute frequency"""
    times = pd.date_range(t0 - pd.Timedelta("1h"),
                        t0 + pd.Timedelta("1h"),
                        freq="5min")
    return xr.DataArray(
        np.arange(len(times)),
        coords={"time_utc": times},
        dims=["time_utc"],
        name="test_data",
    )


def test_dropout_sampling(t0, da_sample):
    """Verify all specified dropout offsets are used when dropout_frac=1"""
    test_deltas = [pd.Timedelta(minutes=-30), pd.Timedelta(minutes=-60)]
    observed = set()

    for _ in range(50):
        result = simulate_dropout(
            da_sample,
            t0,
            dropout_timedeltas=test_deltas,
            dropout_frac=1,
        )
        valid_times = result.where(~xr.ufuncs.isnan(result), drop=True).time_utc
        if valid_times.size > 0:
            # Convert numpy datetime to pandas Timestamp explicitly
            last_valid = pd.Timestamp(valid_times[-1].item())
            offset = (last_valid - t0).total_seconds() / 60
            observed.add(offset)

    assert observed == {-30.0, -60.0}, "All test deltas should be observed"


def test_no_dropout_cases(t0, da_sample):
    """Validate proper handling of no-dropout scenarios"""
    # Case 1: dropout_frac=0 with non-empty deltas
    result = simulate_dropout(
        da_sample,
        t0,
        dropout_timedeltas=[pd.Timedelta("-30min")],
        dropout_frac=0,
    )
    xr.testing.assert_equal(result, da_sample)

    # Case 2: empty deltas with frac=0
    result = simulate_dropout(da_sample, t0, [], dropout_frac=0)
    xr.testing.assert_equal(result, da_sample)

    # Case 3: invalid empty deltas with frac>0
    with pytest.raises(ValueError, match="Must provide dropout_timedeltas"):
        simulate_dropout(da_sample, t0, [], dropout_frac=0.5)


@pytest.mark.parametrize("deltas,expected_mask", [
    ([], False),  # No dropout
    ([pd.Timedelta(0)], True),  # Mask after t0
    ([pd.Timedelta("-30min")], True),  # Mask after t0-30min
])
def test_dropout_application(t0, da_sample, deltas, expected_mask):
    """Verify correct masking behavior for different dropout configurations"""
    result = simulate_dropout(
        da_sample,
        t0,
        dropout_timedeltas=deltas,
        dropout_frac=1 if deltas else 0,
    )

    if expected_mask:
        # Calculate expected cutoff time
        cutoff = t0 + deltas[0]

        # Verify all post-cutoff times are masked
        post_cutoff = da_sample.time_utc > cutoff
        assert result.where(post_cutoff).isnull().all()

        # Verify pre-cutoff times are preserved
        pre_cutoff = da_sample.time_utc <= cutoff
        xr.testing.assert_equal(
            result.where(pre_cutoff, drop=True),
            da_sample.where(pre_cutoff, drop=True),
        )
    else:
        xr.testing.assert_equal(result, da_sample)


def test_input_validation(t0, da_sample):
    """Ensure proper validation of input parameters"""
    # Test positive timedelta (should fail validation)
    with pytest.raises(ValueError, match="must be ≤ 0"):
        simulate_dropout(
            da_sample,
            t0,
            dropout_timedeltas=[pd.Timedelta(hours=1)],  # Explicit positive delta
            dropout_frac=1,
        )

    # Test invalid dropout fraction
    with pytest.raises(ValueError, match="between 0 and 1"):
        simulate_dropout(
            da_sample,
            t0,
            dropout_timedeltas=[pd.Timedelta(minutes=-30)],
            dropout_frac=1.5,  # Invalid fraction
        )

    # Test empty deltas with frac>0
    with pytest.raises(ValueError, match="Must provide dropout_timedeltas"):
        simulate_dropout(
            da_sample,
            t0,
            dropout_timedeltas=[],  # Empty list
            dropout_frac=0.5,
        )


def test_edge_case_handling(t0):
    """Verify correct handling of boundary conditions"""
    # Create test data with exact cutoff alignment
    cutoff = t0 + pd.Timedelta("-30min")
    edge_times = [
        cutoff - pd.Timedelta("5min"),
        cutoff,
        cutoff + pd.Timedelta("5min"),
    ]
    da_edge = xr.DataArray([1, 2, 3], coords={"time_utc": edge_times})

    result = simulate_dropout(
        da_edge,
        t0,
        dropout_timedeltas=[pd.Timedelta("-30min")],
        dropout_frac=1,
    )

    # Verify values
    assert result.sel(time_utc=cutoff).item() == 2  # Exact cutoff preserved
    assert result.sel(time_utc=cutoff + pd.Timedelta("5min")).isnull()
    assert not result.sel(time_utc=cutoff - pd.Timedelta("5min")).isnull()


def test_temporal_alignment(t0, da_sample):
    """Ensure dropout respects temporal relationships"""
    test_delta = pd.Timedelta("-15min")
    result = simulate_dropout(
        da_sample,
        t0,
        dropout_timedeltas=[test_delta],
        dropout_frac=1,
    )

    # Calculate expected valid times
    cutoff = t0 + test_delta
    valid_times = da_sample.time_utc[da_sample.time_utc <= cutoff]
    invalid_times = da_sample.time_utc[da_sample.time_utc > cutoff]

    # Verify valid time preservation
    xr.testing.assert_equal(
        result.sel(time_utc=valid_times),
        da_sample.sel(time_utc=valid_times),
    )

    # Verify invalid time masking
    if invalid_times.size > 0:
        assert result.sel(time_utc=invalid_times).isnull().all()
