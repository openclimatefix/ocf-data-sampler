"""UKV provider loaders."""

import xarray as xr

from ocf_data_sampler.config.model import NWP
from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


def open_ukv(nwp_config: NWP) -> xr.DataArray:
    """Opens the NWP data.

    Args:
        nwp_config: NWP configuration object

    Returns:
        Xarray DataArray of the NWP data
    """
    ds = open_zarr_paths(nwp_config)

    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
            "x": "x_osgb",
            "y": "y_osgb",
        },
    )

    check_time_unique_increasing(ds.init_time_utc)

    ds = make_spatial_coords_increasing(ds, x_coord="x_osgb", y_coord="y_osgb")

    ds = ds.transpose("init_time_utc", "step", "channel", "x_osgb", "y_osgb")

    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
