from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.load.locations import open_locations


def test_open_locations(locations_zarr_path):
    """Test the locations data loader with valid data."""
    ds = open_locations(locations_zarr_path)

    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"longitude", "latitude"}
    assert ds["longitude"].dims == ("location_id",)
    assert ds["latitude"].dims == ("location_id",)
    assert len(np.unique(ds.coords["location_id"])) == ds.sizes["location_id"]


def test_open_locations_missing_data_var(tmp_path: Path):
    """Test that open_locations raises a ValueError when a required data variable is missing."""
    zarr_path = tmp_path / "bad_locations.zarr"

    bad_ds = xr.Dataset(
        data_vars={
            "longitude": (("location_id",), [0.0, 1.0]),
        },
        coords={
            "location_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(ValueError, match="Locations data should have variables"):
        open_locations(zarr_path=str(zarr_path))


def test_open_locations_bad_dtype(tmp_path: Path):
    """Test that open_locations raises a TypeError on incorrect data dtypes."""
    zarr_path = tmp_path / "bad_locations.zarr"

    # Create dataset where longitude is integer
    bad_ds = xr.Dataset(
        data_vars={
            "longitude": (("location_id",), [0, 1]),
            "latitude": (("location_id",), [0.0, 1.0]),
        },
        coords={
            "location_id": [1, 2],
        },
    )
    bad_ds.to_zarr(zarr_path)

    with pytest.raises(TypeError, match="longitude in locations data should be floating"):
        open_locations(zarr_path=str(zarr_path))
