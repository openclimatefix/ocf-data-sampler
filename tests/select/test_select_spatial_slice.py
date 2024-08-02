import numpy as np
import xarray as xr
from ocf_datapipes.utils import Location

from ocf_data_sampler.select.select_spatial_slice import select_spatial_slice_pixels


def test_select_spatial_slice_pixels():
    # Create dummy data
    x = np.arange(100)
    y = np.arange(100)[::-1]

    da = xr.DataArray(
        np.random.normal(size=(len(x), len(y))),
        coords=dict(
            x_osgb=(["x_osgb"], x),
            y_osgb=(["y_osgb"], y),
        )
    )

    location = Location(x=10, y=10, coordinate_system="osgb")

    # Select window which lies within data
    da_sliced = select_spatial_slice_pixels(
        da,
        location,
        width_pixels=10,
        height_pixels=10,
        allow_partial_slice=True,
    )


    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(5, 15)).all()
    assert (da_sliced.y_osgb.values == np.arange(15, 5, -1)).all()
    assert not da_sliced.isnull().any()


    # Select window where the edge of the window lies at the edge of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location,
        width_pixels=20,
        height_pixels=20,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(0, 20)).all()
    assert (da_sliced.y_osgb.values == np.arange(20, 0, -1)).all()
    assert not da_sliced.isnull().any()

    # Select window which is partially outside the boundary of the data
    da_sliced = select_spatial_slice_pixels(
        da,
        location,
        width_pixels=30,
        height_pixels=30,
        allow_partial_slice=True,
    )

    assert isinstance(da_sliced, xr.DataArray)
    assert (da_sliced.x_osgb.values == np.arange(-5, 25)).all()
    assert (da_sliced.y_osgb.values == np.arange(25, -5, -1)).all()
    assert da_sliced.isnull().sum() == 275



