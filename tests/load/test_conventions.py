import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.load.conventions import validate_coords


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
