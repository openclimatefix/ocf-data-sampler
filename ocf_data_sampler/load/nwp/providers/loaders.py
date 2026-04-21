"""NWP provider loaders.

All providers follow the same shape:
    open zarr -> normalise dim/coord names -> shared post-processing.

`_open_regular_grid_nwp` is the shared tail. Per-provider functions only
handle the open + renaming step that differs between data sources.
"""

import logging

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)

_log = logging.getLogger(__name__)


def _open_regular_grid_nwp(
    ds: xr.Dataset | xr.DataArray,
    x_coord: str,
    y_coord: str,
) -> xr.DataArray:
    """Shared post-processing for any regular-grid NWP dataset.

    Expects dims/coords already normalised to: init_time_utc, step, channel,
    plus the given x_coord/y_coord spatial dims.
    """
    check_time_unique_increasing(ds.init_time_utc)
    ds = make_spatial_coords_increasing(ds, x_coord=x_coord, y_coord=y_coord)
    ds = ds.transpose("init_time_utc", "step", "channel", x_coord, y_coord)

    if isinstance(ds, xr.Dataset):
        return get_xr_data_array_from_xr_dataset(ds)
    return ds


def open_ifs(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens ECMWF IFS / MetOffice Global NWP data."""
    ds = open_zarr_paths(zarr_path, backend="tensorstore")
    # LEGACY SUPPORT - older zarrs use "init_time"/"variable" dim names
    ds = ds.rename({"init_time": "init_time_utc", "variable": "channel"})
    return _open_regular_grid_nwp(ds, x_coord="longitude", y_coord="latitude")


def open_gdm(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens GDM (e.g. GenCast) NWP data."""
    ds = open_zarr_paths(zarr_path, backend="tensorstore", time_dim="init_time_utc")
    return _open_regular_grid_nwp(ds, x_coord="longitude", y_coord="latitude")


def open_gfs(zarr_path: str | list[str], public: bool = False) -> xr.DataArray:
    """Opens GFS NWP data."""
    _log.info("Loading NWP GFS data")
    ds = open_zarr_paths(
        zarr_path,
        time_dim="init_time_utc",
        public=public,
        backend="dask",
    )
    nwp = ds.to_array(dim="channel")
    del ds
    return _open_regular_grid_nwp(nwp, x_coord="longitude", y_coord="latitude")


def open_icon_eu(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens DWD ICON-EU data.

    ICON-EU is expected to be on a regular lat/lon grid with a 'channel' dim.
    Only the first 78 (one-hour) steps are used; the rest are 3-hour steps.
    """
    ds = open_zarr_paths(zarr_path, time_dim="init_time_utc", backend="dask")
    if "icon_eu_data" not in ds.data_vars:
        raise ValueError("Could not find 'icon_eu_data' DataArray in the ICON-EU Zarr file.")
    nwp = ds["icon_eu_data"].isel(step=slice(0, 78))
    return _open_regular_grid_nwp(nwp, x_coord="longitude", y_coord="latitude")


def open_ukv(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens UKV NWP data (OSGB grid)."""
    ds = open_zarr_paths(zarr_path, backend="tensorstore")
    # Only rename keys actually present - new UKV data already uses the target names
    rename_map = {
        "init_time": "init_time_utc",
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.coords})
    return _open_regular_grid_nwp(ds, x_coord="x_osgb", y_coord="y_osgb")


def open_cloudcasting(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens OCF cloudcasting satellite-prediction data (geostationary grid).

    References:
        [1] https://www.openclimatefix.org/projects/cloud-forecasting
        [2] https://github.com/ClimeTrend/cloudcasting
        [3] https://github.com/openclimatefix/sat_pred
    """
    ds = open_zarr_paths(zarr_path, backend="tensorstore")
    ds = ds.rename({"init_time": "init_time_utc", "variable": "channel"})
    return _open_regular_grid_nwp(ds, x_coord="x_geostationary", y_coord="y_geostationary")
