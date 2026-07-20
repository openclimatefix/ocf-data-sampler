"""Module for opening NWP data."""


from collections.abc import Mapping

import numpy as np
import xarray as xr

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.common.xr_tensorstore import ZarrSource, open_zarr_paths
from ocf_data_sampler.load.conventions import (
    extract_single_data_array,
    make_spatial_coords_increasing,
    validate_coords,
)

SpatialCoords = tuple[str, str]

# This is our central registry of NWP providers and their spatial coordinate names, which allows us
# to handle different providers in a consistent way.
PROVIDER_REGISTRY: dict[str, SpatialCoords] = {
    "ukv": ("x_osgb", "y_osgb"),
    "ecmwf": ("longitude", "latitude"),
    "mo_global": ("longitude", "latitude"),
    "gencast": ("longitude", "latitude"),
    "fgn": ("longitude", "latitude"),
    "cloudcasting": ("x_geostationary", "y_geostationary"),
}


def open_nwp(zarr_path: ZarrSource, provider: str) -> xr.DataArray:
    """Open NWP data and conform it to the standard layout."""
    provider = provider.lower()

    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider: {provider!r}. Supported: {supported}")

    x_coord, y_coord = PROVIDER_REGISTRY[provider]

    ds = open_zarr_paths(zarr_path, concat_dim="init_time_utc")
    ds = _rename(ds, name_mapping={"variable": "channel"})

    expected_coord_dtypes = {
        "init_time_utc": np.datetime64,
        "step": np.timedelta64,
        "channel": np.str_,
        x_coord: np.number,
        y_coord: np.number,
    }
    validate_coords(
        ds,
        expected_coord_dtypes,
        source=f"NWP provider {provider!r}",
    )

    ds = make_spatial_coords_increasing(ds, x_coord=x_coord, y_coord=y_coord)

    assert_values_unique_increasing(ds["init_time_utc"].values, "init_time_utc")
    assert_values_unique_increasing(ds["step"].values, "step")

    da = extract_single_data_array(ds)
    da = da.transpose("init_time_utc", "step", "channel", x_coord, y_coord)

    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"NWP data for {provider} should be floating, not {da.dtype}")

    return da


def _rename(ds: xr.Dataset, name_mapping: Mapping[str, str]) -> xr.Dataset:
    """Renames coordinates in the dataset based on the provided mapping.

    Args:
        ds: The xarray.Dataset to rename.
        name_mapping: A dictionary mapping old variable names to new names.
    """
    # Only rename variables that are actually present in the dataset
    return ds.rename({k: v for k, v in name_mapping.items() if k in ds})
