"""Cloudcasting provider loader."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_cloudcasting(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens the satellite predictions from cloudcasting.

    Cloudcasting is a OCF forecast product. We forecast future satellite images from recent
    satellite images. More information can be found in the references below.

    Args:
        zarr_path: Path to the zarr(s) to open

    Returns:
        Xarray DataArray of the cloudcasting data

    References:
            [1] https://www.openclimatefix.org/projects/cloud-forecasting
            [2] https://github.com/ClimeTrend/cloudcasting
            [3] https://github.com/openclimatefix/sat_pred
    """
    # Open the data
    ds = open_zarr_paths(zarr_path)

    # Rename
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        },
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.init_time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")

    ds = ds.transpose("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")

    return get_xr_data_array_from_xr_dataset(ds)
