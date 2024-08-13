import xarray as xr
import pandas as pd

def check_time_unique_increasing(datetimes) -> None:
    """Check that the time dimension is unique and increasing"""
    time = pd.DatetimeIndex(datetimes)
    assert time.is_unique
    assert time.is_monotonic_increasing

def make_spatial_coords_increasing(ds: xr.Dataset, x_coord: str, y_coord: str) -> xr.Dataset:
    """Make sure the spatial coordinates are in increasing order
    
    Args:
        ds: Xarray Dataset
        x_coord: Name of the x coordinate
        y_coord: Name of the y coordinate
    """

    # Make sure the coords are in increasing order
    if ds[x_coord][0] > ds[x_coord][-1]:
        ds = ds.isel({x_coord:slice(None, None, -1)})
    if ds[y_coord][0] > ds[y_coord][-1]:
       ds = ds.isel({y_coord:slice(None, None, -1)})

    # Check the coords are all increasing now
    assert (ds[x_coord].diff(dim=x_coord) > 0).all()
    assert (ds[y_coord].diff(dim=y_coord) > 0).all()

    return ds