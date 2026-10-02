import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.load.conventions import (
    validate_coords,
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
        np.timedelta64(resolution_minutes, "m"),
        source="test data",
    )


def test_validate_time_grid_rejects_misaligned_time():
    times = np.array(["2024-01-01T00:00", "2024-01-01T00:06"], dtype="datetime64[ns]")

    with pytest.raises(
        ValueError,
        match=r"test data: timestamp 2024-01-01T00:06:00.*configured resolution 5 minutes",
    ):
        validate_time_grid(times, np.timedelta64(5, "m"), source="test data")


def test_validate_step_grid_accepts_aligned_steps():
    validate_step_grid(
        np.array([0, 60, 180], dtype="timedelta64[m]"),
        np.timedelta64(60, "m"),
        source="nwp/test",
    )


def test_validate_step_grid_rejects_misaligned_step():
    steps = np.array([0, 60, 90], dtype="timedelta64[m]")

    with pytest.raises(
        ValueError,
        match="nwp/test: step 90 minutes is not a multiple of the configured resolution 60 minutes",
    ):
        validate_step_grid(steps, np.timedelta64(60, "m"), source="nwp/test")
