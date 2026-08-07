"""Satellite loader."""

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.common.xr_tensorstore import ZarrSource, open_zarr_paths
from ocf_data_sampler.load.conventions import (
    extract_single_data_array,
    make_spatial_coords_increasing,
    validate_coords,
)


def open_sat_data(zarr_path: ZarrSource) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types.

    Args:
        zarr_path: Path(s) to the zarr file(s)
    """
    ds = open_zarr_paths(zarr_path, concat_dim="time_utc")

    coord_dtypes = {
        "time_utc": np.datetime64,
        "channel": np.str_,
        "x_geostationary": np.number,
        "y_geostationary": np.number,
    }
    validate_coords(
        ds,
        coord_dtypes,
        source="satellite data",
    )

    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    assert_values_unique_increasing(ds["time_utc"].values, "time_utc")

    da = extract_single_data_array(ds)
    da = da.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")

    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"Satellite data should be floating, not {da.dtype}")

    return da
