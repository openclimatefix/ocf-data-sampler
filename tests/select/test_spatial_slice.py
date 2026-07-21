import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.select.spatial_slice import (
    _get_central_index,
    _get_window_bounds,
    select_spatial_slice_pixels,
)
from ocf_data_sampler.spatial import Location


@pytest.fixture(scope="module")
def da():
    """Create dummy 2D spatial data"""
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    return xr.DataArray(
        np.random.normal(size=(len(x), len(y))),
        coords={
            "x_osgb": (["x_osgb"], x),
            "y_osgb": (["y_osgb"], y),
        },
    )

@pytest.mark.parametrize(
    "x_query, expected_xval, method",
    [
        (4.1, 4, "nearest"),
        (3.9, 4, "nearest"),
        (4, 4, "nearest"),
        (4.1, 4, "left"),
        (3.9, 3, "left"),
        (4, 4, "left"),
    ],
)
def test_get_central_index(x_query, expected_xval, method):
    x_vals = np.arange(0, 10)
    index = _get_central_index(x_vals, x_query, method=method)
    assert x_vals[index] == expected_xval


@pytest.mark.parametrize(
    "central_index, window_size, expected_slice",
    [
        (5, 2, (5, 7)),
        (5, 3, (4, 7)),
        (5, 4, (4, 8)),
        (5, 5, (3, 8)),
    ]
)
def test_get_window_bounds(central_index, window_size, expected_slice):
    index_slice = _get_window_bounds(central_index, window_size)
    assert index_slice == expected_slice


def test_select_spatial_slice_pixels(da):

    # Select odd sized window
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=10.1, y=-4.9, coord_system="osgb"),
        width_pixels=3,
        height_pixels=3,
    )

    assert (da_sliced["x_osgb"].values == np.array([9, 10, 11])).all()
    assert (da_sliced["y_osgb"].values == np.array([-6, -5, -4])).all()
    assert not da_sliced.isnull().any()


    # Select even sized window
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=10.1, y=-4.9, coord_system="osgb"),
        width_pixels=4,
        height_pixels=4,
    )

    assert (da_sliced["x_osgb"].values == np.array([9, 10, 11, 12])).all()
    assert (da_sliced["y_osgb"].values == np.array([-6, -5, -4, -3])).all()
    assert not da_sliced.isnull().any()


    # Select mixed odd and even sized window
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=10.1, y=-4.9, coord_system="osgb"),
        width_pixels=3,
        height_pixels=4,
    )

    assert (da_sliced["x_osgb"].values == np.array([9, 10, 11])).all()
    assert (da_sliced["y_osgb"].values == np.array([-6, -5, -4, -3])).all()
    assert not da_sliced.isnull().any()

    # Select window where the edge of the window lies right on the edge of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90.1, y=89.9, coord_system="osgb"),
        width_pixels=20,
        height_pixels=20,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced["x_osgb"].values == np.arange(-100, -80)).all()
    assert (da_sliced["y_osgb"].values == np.arange(80, 100)).all()
    assert not da_sliced.isnull().any()


def test_select_spatial_slice_pixels_out_of_bounds(da):
    """Test that ValueError is raised when the requested slice goes out of bounds."""
    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=-90.1, y=-80.1, coord_system="osgb"),
            width_pixels=30,
            height_pixels=30,
        )
    msg = str(excinfo.value)
    assert "Slice is unavailable:" in msg

    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=90.1, y=90.1, coord_system="osgb"),
            width_pixels=40,
            height_pixels=40,
        )
    msg = str(excinfo.value)
    assert "Slice is unavailable:" in msg
