"""Module for opening NWP data."""


from glob import glob
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.common.xr_tensorstore import open_zarr, open_zarrs
from ocf_data_sampler.load.conventions import (
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def open_zarr_paths(
    zarr_path: str | list[str],
    time_dim: str,
    backend: Literal["dask", "tensorstore"],
    public: bool = False,
) -> xr.Dataset:
    """Opens the NWP data.

    Args:
        zarr_path: Path to the zarr(s) to open
        time_dim: Name of the time dimension
        public: Whether the data is public or private. Only available for the dask backend.
        backend: The xarray backend to use.

    Returns:
        The opened Xarray Dataset
    """
    if backend not in ["dask", "tensorstore"]:
        raise ValueError(
            f"Unsupported backend: {backend}. Supported backends are 'dask' and 'tensorstore'.",
        )

    if public and backend == "tensorstore":
        raise ValueError("Public data is only supported with the 'dask' backend.")

    if backend == "tensorstore":
        ds = _tensorstore_open_zarr_paths(zarr_path, time_dim)

    elif backend == "dask":
        ds = _dask_open_zarr_paths(zarr_path, time_dim, public)

    return ds


def _dask_open_zarr_paths(zarr_path: str | list[str], time_dim: str, public: bool) -> xr.Dataset:
    general_kwargs = {
        "engine": "zarr",
        "chunks": "auto",
        "decode_timedelta": True,
    }

    if public:
        # note this only works for s3 zarr paths at the moment
        general_kwargs["storage_options"] = {"anon": True}

    if isinstance(zarr_path, list | tuple) or "*" in str(zarr_path):  # Multi-file dataset
        ds = xr.open_mfdataset(
            zarr_path,
            concat_dim=time_dim,
            combine="nested",
            coords="different",
            compat="no_conflicts",
            **general_kwargs,
        ).sortby(time_dim)
    else:
        ds = xr.open_dataset(
            zarr_path,
            consolidated=True,
            mode="r",
            **general_kwargs,
        )
    return ds


def _tensorstore_open_zarr_paths(zarr_path: str | list[str], time_dim: str) -> xr.Dataset:

    if "*" in str(zarr_path):
        zarr_path = sorted(glob(zarr_path))

    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim=time_dim, data_source="nwp").sortby(time_dim)
    else:
        ds = open_zarr(zarr_path)
    return ds


def _validate_nwp_data(data_array: xr.DataArray, provider: str) -> None:
    """Validates the structure and data types of a loaded NWP DataArray.

    This helper function is extracted to keep the main `open_nwp` function clean.

    Args:
        data_array: The xarray.DataArray to validate.
        provider: The NWP provider name.

    Raises:
        TypeError: If the data or any coordinate has an unexpected dtype.
        ValueError: If a required coordinate is missing.
    """
    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError(
            f"NWP data for {provider} should be numeric, not {data_array.dtype}",
        )

    common_expected_dtypes = {
        "init_time_utc": np.datetime64,
        "step": np.timedelta64,
    }

    geographic_spatial_dtypes = {
        "latitude": np.floating,
        "longitude": np.floating,
    }

    provider_specific_spatial_dtypes = {
        "ecmwf": geographic_spatial_dtypes,
        "icon-eu": geographic_spatial_dtypes,
        "gfs": geographic_spatial_dtypes,
        "mo_global": geographic_spatial_dtypes,
        "ukv": {
            "x_osgb": np.number,
            "y_osgb": np.number,
        },
        "cloudcasting": {
            "x_geostationary": np.floating,
            "y_geostationary": np.floating,
        },
    }

    expected_dtypes = {
        **common_expected_dtypes,
        **provider_specific_spatial_dtypes.get(provider, {}),
    }

    for coord, expected_dtype in expected_dtypes.items():
        if coord not in data_array.coords:
            raise ValueError(f"Coordinate '{coord}' missing for provider '{provider}'")

        actual_dtype = data_array.coords[coord].dtype

        if not np.issubdtype(actual_dtype, expected_dtype):
            if isinstance(expected_dtype, tuple):
                expected_name_str = " or ".join([t.__name__ for t in expected_dtype])
            else:
                expected_name_str = expected_dtype.__name__

            err_msg = (
                f"'{coord}' for {provider} should be {expected_name_str}, "
                f"not {actual_dtype.name}"
            )
            raise TypeError(err_msg)


def open_nwp(
    zarr_path: str | list[str],
    provider: str,
    public: bool = False,
) -> xr.DataArray:
    """Opens NWP zarr and validates its structure and data types.

    Args:
        zarr_path: path to the zarr file
        provider: NWP provider
        public: Whether the data is public or private (only for GFS)
    """
    provider = provider.lower()

    _OPEN_NWP_FUNCTIONS: dict[str, Callable[..., xr.DataArray]] = {
        "ukv": open_ukv,
        "ecmwf": open_standard_lon_lat_grid,
        "mo_global": open_standard_lon_lat_grid,
        "icon-eu": open_icon_eu,
        "gencast": open_standard_lon_lat_grid,
        "fgn": open_standard_lon_lat_grid,
        "gfs": open_gfs,
        "cloudcasting": open_cloudcasting,
    }

    if provider not in _OPEN_NWP_FUNCTIONS:
        supported = ", ".join(sorted(_OPEN_NWP_FUNCTIONS.keys()))
        raise ValueError(f"Unknown provider: {provider!r}. Supported: {supported}")

    opener = _OPEN_NWP_FUNCTIONS[provider]

    kwargs = {"zarr_path": zarr_path}
    if provider == "gfs" and public:
        kwargs["public"] = True

    data_array = opener(**kwargs)
    _validate_nwp_data(data_array, provider)
    return data_array


def _canonicalize_regular_grid_layout(
    ds: xr.Dataset | xr.DataArray,
    x_coord: str,
    y_coord: str,
) -> xr.DataArray:
    """Shared post-processing for any regular-grid NWP dataset.

    Expects dims/coords already standardised to: init_time_utc, step, channel,
    plus the given x_coord/y_coord spatial dims.
    """
    assert_values_unique_increasing(ds["init_time_utc"].values, "init_time_utc")
    ds = make_spatial_coords_increasing(ds, x_coord=x_coord, y_coord=y_coord)
    ds = ds.transpose("init_time_utc", "step", "channel", x_coord, y_coord)

    if isinstance(ds, xr.Dataset):
        return get_xr_data_array_from_xr_dataset(ds)
    return ds


def open_standard_lon_lat_grid(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens NWP data on a standard latitude/longitude grid.

    Used by ECMWF IFS, MetOffice Global, GDM (e.g. GenCast & FGN).
    Pass time_dim="init_time_utc" for zarrs that already use the new dim name.
    """
    ds = open_zarr_paths(zarr_path, backend="tensorstore", time_dim="init_time_utc")
    rename_map = {"variable": "channel"}
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.coords})
    return _canonicalize_regular_grid_layout(ds, x_coord="longitude", y_coord="latitude")


def open_gfs(zarr_path: str | list[str], public: bool = False) -> xr.DataArray:
    """Opens GFS NWP data."""
    ds = open_zarr_paths(
        zarr_path,
        time_dim="init_time_utc",
        public=public,
        backend="dask",
    )
    da = ds.to_array(dim="channel")
    return _canonicalize_regular_grid_layout(da, x_coord="longitude", y_coord="latitude")


def open_icon_eu(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens DWD ICON-EU data.

    ICON-EU is expected to be on a regular lat/lon grid with a 'channel' dim.
    Only the first 78 (one-hour) steps are used; the rest are 3-hour steps.
    """
    ds = open_zarr_paths(zarr_path, time_dim="init_time_utc", backend="dask")
    if "icon_eu_data" not in ds.data_vars:
        raise ValueError("Could not find 'icon_eu_data' DataArray in the ICON-EU Zarr file.")
    nwp = ds["icon_eu_data"].isel(step=slice(0, 78))
    return _canonicalize_regular_grid_layout(nwp, x_coord="longitude", y_coord="latitude")


def open_ukv(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens UKV NWP data (OSGB grid)."""
    ds = open_zarr_paths(zarr_path, backend="tensorstore", time_dim="init_time_utc")
    # Only rename keys actually present - new UKV data already uses the target names
    rename_map = {
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }
    ds = ds.rename({k: v for k, v in rename_map.items() if k in ds.coords})
    return _canonicalize_regular_grid_layout(ds, x_coord="x_osgb", y_coord="y_osgb")


def open_cloudcasting(zarr_path: str | list[str]) -> xr.DataArray:
    """Opens OCF cloudcasting satellite-prediction data (geostationary grid).

    References:
        [1] https://www.openclimatefix.org/projects/cloud-forecasting
        [2] https://github.com/ClimeTrend/cloudcasting
        [3] https://github.com/openclimatefix/sat_pred
    """
    ds = open_zarr_paths(zarr_path, time_dim="init_time_utc", backend="tensorstore")
    ds = ds.rename({"variable": "channel"})
    return _canonicalize_regular_grid_layout(
        ds, x_coord="x_geostationary", y_coord="y_geostationary",
    )
