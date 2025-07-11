"""Module for opening NWP data."""

import numpy as np
import xarray as xr

from ocf_data_sampler.load.nwp.providers.cloudcasting import open_cloudcasting
from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.gfs import open_gfs
from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv


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
        raise TypeError(f"NWP data for {provider} should be numeric, not {data_array.dtype}")

    common_expected_dtypes = {
        "init_time_utc": np.datetime64,
        "step": np.timedelta64,
        "channel": (np.str_, np.object_),
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

    kwargs = {
        "zarr_path": zarr_path,
    }
    if provider == "ukv":
        _open_nwp = open_ukv
    elif provider in ["ecmwf", "mo_global"]:
        _open_nwp = open_ifs
    elif provider == "icon-eu":
        _open_nwp = open_icon_eu
    elif provider == "gfs":
        _open_nwp = open_gfs
        # GFS has a public/private flag
        if public:
            kwargs["public"] = True
    elif provider == "cloudcasting":
        _open_nwp = open_cloudcasting
    else:
        raise ValueError(f"Unknown provider: {provider}")

    data_array = _open_nwp(**kwargs)
    _validate_nwp_data(data_array, provider)

    return data_array
