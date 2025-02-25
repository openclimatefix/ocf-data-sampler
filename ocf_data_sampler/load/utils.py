"""Utility functions for working with xarray objects."""

import pandas as pd
import xarray as xr

from ocf_data_sampler.numpy_sample.sun_position import calculate_azimuth_and_elevation


def check_time_unique_increasing(datetimes: xr.DataArray) -> None:
    """Check that the time dimension is unique and increasing."""
    time = pd.DatetimeIndex(datetimes)
    if not (time.is_unique and time.is_monotonic_increasing):
        raise ValueError("Time dimension must be unique and monotonically increasing")


def make_spatial_coords_increasing(ds: xr.Dataset, x_coord: str, y_coord: str) -> xr.Dataset:
    """Make sure the spatial coordinates are in increasing order.

    Args:
        ds: Xarray Dataset
        x_coord: Name of the x coordinate
        y_coord: Name of the y coordinate
    """
    # Make sure the coords are in increasing order
    if ds[x_coord][0] > ds[x_coord][-1]:
        ds = ds.isel({x_coord: slice(None, None, -1)})
    if ds[y_coord][0] > ds[y_coord][-1]:
        ds = ds.isel({y_coord: slice(None, None, -1)})

    # Check the coords are all increasing now
    if not (ds[x_coord].diff(dim=x_coord) > 0).all():
        raise ValueError(f"'{x_coord}' coordinate must be increasing")
    if not (ds[y_coord].diff(dim=y_coord) > 0).all():
        raise ValueError(f"'{y_coord}' coordinate must be increasing")

    return ds


def get_xr_data_array_from_xr_dataset(ds: xr.Dataset) -> xr.DataArray:
    """Return underlying xr.DataArray from passed xr.Dataset.

    Checks only one variable is present and returns it as an xr.DataArray.

    Args:
        ds: xr.Dataset to extract xr.DataArray from
    """
    datavars = list(ds.var())
    if len(datavars) != 1:
        raise ValueError("Cannot open as xr.DataArray: dataset contains multiple variables")
    return ds[datavars[0]]


def add_solar_position(
    data_array: xr.DataArray,
    id_dim: str,
    x_attr: str,
    y_attr: str,
) -> xr.DataArray:
    """Add solar position data to DataArray.

    Args:
        data_array: The DataArray to modify
        id_dim: The dimension name for the location IDs (e.g., 'gsp_id' or 'site_id')
        x_attr: The attribute name for longitude (e.g., 'lon_osgb' or 'longitude')
        y_attr: The attribute name for latitude (e.g., 'lat_osgb' or 'latitude')

    Returns:
        The modified DataArray with solar position data added
    """
    # Extract timestamps
    times = pd.DatetimeIndex(data_array.time_utc.values)
    check_time_unique_increasing(times)

    # For each location, calculate and add solar position
    for location_id in data_array[id_dim].values:
        # Get location coordinates
        location_data = data_array.sel({id_dim: location_id})
        lon = float(location_data[x_attr].values)
        lat = float(location_data[y_attr].values)

        # Calculate solar position and normalise
        azimuth, elevation = calculate_azimuth_and_elevation(times, lon, lat)  
        azimuth = azimuth / 360
        elevation = elevation / 180 + 0.5

        # Add as new variables to DataArray
        data_array.loc[{id_dim: location_id}]["solar_azimuth"] = (("time_utc"), azimuth)
        data_array.loc[{id_dim: location_id}]["solar_elevation"] = (("time_utc"), elevation)

    return data_array


def add_solar_position_to_gsp(da_gsp: xr.DataArray) -> xr.DataArray:
    """Add solar position data to GSP DataArray."""
    return add_solar_position(da_gsp, id_dim="gsp_id", x_attr="lon_osgb", y_attr="lat_osgb")


def add_solar_position_to_site(da_site: xr.DataArray) -> xr.DataArray:
    """Add solar position data to Site DataArray."""
    return add_solar_position(da_site, id_dim="site_id", x_attr="longitude", y_attr="latitude")
