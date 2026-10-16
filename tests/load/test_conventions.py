import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.load.conventions import (
    validate_coords,
    validate_matching_projections,
    validate_step_grid,
    validate_time_grid,
)


def test_validate_coords_rejects_multidimensional_coord():
    ds = xr.Dataset(
        coords={
            "latitude": (
                ("x", "y"),
                np.array([[50.0, 51.0], [52.0, 53.0]]),
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="Coordinate 'latitude' in test data should be 1D, not 2D",
    ):
        validate_coords(
            ds,
            {"latitude": np.number},
            source="test data",
        )


@pytest.mark.parametrize(
    "timestamps,resolution_minutes",
    [
        (["2024-01-01T00:00", "2024-01-01T00:05", "2024-01-01T00:15"], 5),
        (["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00"], 60),
    ],
)
def test_validate_time_grid_accepts_aligned_times(timestamps, resolution_minutes):
    validate_time_grid(
        np.array(timestamps, dtype="datetime64[ns]"),
        minutes(resolution_minutes),
        source="test data",
    )


def test_validate_time_grid_rejects_misaligned_time():
    times = np.array(["2024-01-01T00:00", "2024-01-01T00:06"], dtype="datetime64[ns]")

    with pytest.raises(
        ValueError,
        match=r"test data: timestamp 2024-01-01T00:06:00.*configured resolution 5 minutes",
    ):
        validate_time_grid(times, minutes(5), source="test data")


def test_validate_step_grid_accepts_valid_steps():
    validate_step_grid(
        minutes([0, 60, 120]),
        minutes(60),
        source="nwp/test",
    )


@pytest.mark.parametrize("steps", [[0, 60, 90], [0, 180, 360]])
def test_validate_step_grid_rejects_invalid_steps(steps):
    with pytest.raises(
        ValueError,
        match=(
            r"nwp/test: steps .* must be consecutive multiples "
            "of the configured resolution 60 minutes"
        ),
    ):
        validate_step_grid(minutes(steps), minutes(60), source="nwp/test")


def test_validate_matching_projections_rejects_different_projections():
    datasets = [
        xr.Dataset(attrs={"area": {"satellite": {"projection": {"lon_0": longitude}}}})
        for longitude in [9.5, 0.0]
    ]
    with pytest.raises(
        ValueError,
        match=r"Multiple geostationary projections found: \[\{'lon_0': 9.5\}, \{'lon_0': 0.0\}\]",
    ):
        validate_matching_projections(datasets)
