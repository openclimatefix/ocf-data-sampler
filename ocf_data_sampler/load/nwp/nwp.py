"""Module for opening NWP data."""

import numpy as np
import xarray as xr

from ocf_data_sampler.load.nwp.providers.cloudcasting import open_cloudcasting
from ocf_data_sampler.load.nwp.providers.ecmwf import open_ifs
from ocf_data_sampler.load.nwp.providers.gfs import open_gfs
from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv


def open_nwp(
    zarr_path: str | list[str],
    provider: str,
    public: bool = False,
    *,
    _test_data: xr.DataArray = None,
) -> xr.DataArray:
    """Opens NWP zarr and validates its structure and data types.

    Args:
        zarr_path: path to the zarr file
        provider: NWP provider
        public: Whether the data is public or private (only for GFS)
        _test_data: Used internally for testing to inject data without file IO.
    """
    provider = provider.lower()

    if _test_data is not None:
        data_array = _test_data
    else:
        kwargs = {
            "zarr_path": zarr_path,
        }
        if provider == "ukv":
            _open_nwp = open_ukv
        elif provider in ["ecmwf", "mo_global"]:
            provider = "ecmwf"
            _open_nwp = open_ifs
        elif provider == "icon-eu":
            _open_nwp = open_icon_eu
        elif provider == "gfs":
            _open_nwp = open_gfs
            if public:
                kwargs["public"] = True
        elif provider == "cloudcasting":
            _open_nwp = open_cloudcasting
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Load data using provider-specific function
        data_array = _open_nwp(**kwargs)

    # Validate loaded data array
    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError(f"NWP data for {provider} should be numeric, not {data_array.dtype}")

    if provider == "ukv":
        expected_dtypes = {
            "init_time_utc": np.datetime64,
            "step": np.timedelta64,
            "channel": np.str_,
            "x_osgb": np.floating,
            "y_osgb": np.floating,
        }
    elif provider == "ecmwf":
        expected_dtypes = {
            "init_time_utc": np.datetime64,
            "step": np.timedelta64,
            "channel": np.str_,
            "latitude": np.integer,
            "longitude": np.integer,
        }
    elif provider == "icon-eu":
        expected_dtypes = {
            "init_time_utc": np.datetime64,
            "step": np.timedelta64,
            "channel": np.object_,
            "latitude": np.floating,
            "longitude": np.floating,
        }
    elif provider == "cloudcasting":
        expected_dtypes = {
            "init_time_utc": np.datetime64,
            "step": np.timedelta64,
            "channel": np.str_,
            "x_geostationary": np.floating,
            "y_geostationary": np.floating,
        }
    elif provider == "gfs":
        expected_dtypes = {
            "init_time_utc": np.datetime64,
            "step": np.timedelta64,
            "channel": np.str_,
            "latitude": np.floating,
            "longitude": np.floating,
        }
    else:
        expected_dtypes = {}

    for coord, expected_dtype in expected_dtypes.items():
        if coord not in data_array.coords:
            raise ValueError(f"Coordinate '{coord}' missing for provider '{provider}'")
        if not np.issubdtype(data_array.coords[coord].dtype, expected_dtype):
            actual_dtype = data_array.coords[coord].dtype
            err_msg = (
                f"'{coord}' for {provider} should be {expected_dtype.__name__}, "
                f"not {actual_dtype}"
            )
            raise TypeError(err_msg)

    return data_array
