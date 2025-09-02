import numpy as np
import pytest
import xarray as xr

from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_spatial_slice import (
    _get_pixel_index_location,  # Corrected import
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
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-95, -85)).all()
    assert (da_sliced.y_osgb.values == np.arange(-85, -75)).all()
    # No padding in this case so no NaNs
    assert not da_sliced.isnull().any()

    # Select window where the edge of the window lies right on the edge of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=-80, coord_system="osgb"),
        width_pixels=20,
        height_pixels=20,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-100, -80)).all()
    assert (da_sliced.y_osgb.values == np.arange(-90, -70)).all()
    # No padding in this case so no NaNs
    assert not da_sliced.isnull().any()

    # Select window which is partially outside the boundary of the data - padded on left
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=-80, coord_system="osgb"),
        width_pixels=30,
        height_pixels=30,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-105, -75)).all()
    assert (da_sliced.y_osgb.values == np.arange(-95, -65)).all()
    # Data has been padded on left by 5 NaN pixels
    assert da_sliced.isnull().sum() == 5 * len(da_sliced.y_osgb)

    # Select window which is partially outside the boundary of the data - padded on right
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=90, y=-80, coord_system="osgb"),
        width_pixels=30,
        height_pixels=30,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(75, 105)).all()
    assert (da_sliced.y_osgb.values == np.arange(-95, -65)).all()
    # Data has been padded on right by 5 NaN pixels
    assert da_sliced.isnull().sum() == 5 * len(da_sliced.y_osgb)

    # Select window which is partially outside the boundary of the data - padded on top
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=95, coord_system="osgb"),
        width_pixels=20,
        height_pixels=20,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-100, -80)).all()
    assert (da_sliced.y_osgb.values == np.arange(85, 105)).all()
    # Data has been padded on top by 5 NaN pixels
    assert da_sliced.isnull().sum() == 5 * len(da_sliced.x_osgb)

    # Select window which is partially outside the boundary of the data - padded on bottom
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=-90, y=-95, coord_system="osgb"),
        width_pixels=20,
        height_pixels=20,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-100, -80)).all()
    assert (da_sliced.y_osgb.values == np.arange(-105, -85)).all()
    # Data has been padded on bottom by 5 NaN pixels
    assert da_sliced.isnull().sum() == 5 * len(da_sliced.x_osgb)

    # Select window which is partially outside the boundary of the data - padded right and bottom
    da_sliced = select_spatial_slice_pixels(
        da,
        location=Location(x=90, y=-80, coord_system="osgb"),
        width_pixels=50,
        height_pixels=50,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(65, 115)).all()
    assert (da_sliced.y_osgb.values == np.arange(-105, -55)).all()
    # Data has been padded on right by 15 pixels and bottom by 5 NaN pixels
    assert (
        da_sliced.isnull().sum() == 15 * len(da_sliced.y_osgb) + 5 * len(da_sliced.x_osgb) - 15 * 5
    )


def test_select_spatial_slice_pixels_no_partial(da):
    """Test that ValueError is raised when allow_partial_slice=False and padding is needed."""
    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=-90, y=-80, coord_system="osgb"),
            width_pixels=30,
            height_pixels=30,
            allow_partial_slice=False,
        )
    msg = str(excinfo.value)
    assert "Padding required" in msg
    assert "left_idx" in msg or "bottom_idx" in msg or "right_idx" in msg or "top_idx" in msg

    with pytest.raises(ValueError) as excinfo:
        select_spatial_slice_pixels(
            da,
            location=Location(x=90, y=90, coord_system="osgb"),
            width_pixels=40,
            height_pixels=40,
            allow_partial_slice=False,
        )
    msg = str(excinfo.value)
    assert "Padding required" in msg
    assert "left_idx" in msg or "bottom_idx" in msg or "right_idx" in msg or "top_idx" in msg
