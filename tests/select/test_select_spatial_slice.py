import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_spatial_slice import (
    _get_pixel_index_location,
    select_spatial_slice_pixels,
)


@pytest.fixture(scope="module")
def da():
    # Create dummy data
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)

    da = xr.DataArray(
        np.random.normal(size=(len(x), len(y))),
        coords={
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )
    return da


def test_get_idx_of_pixel_closest_to_poi(da):
    idx_location = _get_pixel_index_location(da, location=Location(x=10, y=10, coord_system="osgb"))
    assert idx_location == (110, 110)


def test_select_spatial_slice_pixels(da):
    # Select window which lies within x-y bounds of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=-80, coord_system="osgb"),
        width_pixels=10,
        height_pixels=10,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-95, -85)).all()
    assert (da_sliced.y_osgb.values == np.arange(-85, -75)).all()
    assert not da_sliced.isnull().any()

    # Select window where the edge of the window lies right on the edge of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=-80, coord_system="osgb"),
        width_pixels=20,
        height_pixels=20,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-100, -80)).all()
    assert (da_sliced.y_osgb.values == np.arange(-90, -70)).all()
    assert not da_sliced.isnull().any()


def test_select_spatial_slice_pixels_out_of_bounds(da):
    """Test that ValueError is raised when the requested slice goes out of bounds."""
    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=-90, y=-80, coord_system="osgb"),
            width_pixels=30,
            height_pixels=30,
        )
    msg = str(excinfo.value)
    assert "not available" in msg

    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=90, y=90, coord_system="osgb"),
            width_pixels=40,
            height_pixels=40,
        )
    msg = str(excinfo.value)
    assert "not available" in msg
