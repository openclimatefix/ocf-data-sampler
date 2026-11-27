"""Module for opening NWP data."""

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import assert_values_unique_increasing
from ocf_data_sampler.common.types import ZarrSource
from ocf_data_sampler.common.xr_tensorstore import open_zarr_paths
from ocf_data_sampler.load.conventions import (
    extract_only_data_array,
    make_spatial_coords_increasing,
    validate_coords,
    validate_matching_projections,
)
from ocf_data_sampler.spatial import detect_coordinate_system


def open_nwp(zarr_path: ZarrSource) -> xr.DataArray:
    """Open NWP data and conform it to the standard layout."""
    ds = open_zarr_paths(zarr_path, concat_dim="init_time_utc", validator=_validate_stores)
    _, x_coord, y_coord = detect_coordinate_system(ds)
    ds = _rename_coords(ds, name_mapping={"variable": "channel"})

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
        source="NWP data",
    )

    ds = make_spatial_coords_increasing(ds, x_coord=x_coord, y_coord=y_coord)

    assert_values_unique_increasing(ds["init_time_utc"].values, "init_time_utc")
    assert_values_unique_increasing(ds["step"].values, "step")
    _assert_steps_uniformly_spaced(ds["step"].values)

    da = extract_only_data_array(ds)
    da = da.transpose("init_time_utc", "step", "channel", x_coord, y_coord)

    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"NWP data should be floating, not {da.dtype}")

    return da


def _validate_stores(datasets: Sequence[xr.Dataset]) -> None:
    """If the stores use geostationary coordinates, require them to share one projection."""
    coord_system, _, _ = detect_coordinate_system(datasets[0])
    if coord_system == "geostationary":
        validate_matching_projections(datasets)


def _assert_steps_uniformly_spaced(steps: NDArray[np.timedelta64]) -> None:
    """Assert that the forecast steps are evenly spaced.

    Args:
        steps: The forecast steps, already checked to be unique and increasing.
    """
    spacings = np.unique(np.diff(steps))

    if len(spacings) > 1:
        raise ValueError(
            "NWP steps must be evenly spaced, but the steps are "
            f"spaced by {spacings}. Mixed step spacing is not supported.",
        )


def _rename_coords(ds: xr.Dataset, name_mapping: Mapping[str, str]) -> xr.Dataset:
    """Renames coordinates in the dataset based on the provided mapping.

    Args:
        ds: The xarray.Dataset to rename.
        name_mapping: A dictionary mapping old variable names to new names.
    """
    # Only rename variables that are actually present in the dataset
    return ds.rename({k: v for k, v in name_mapping.items() if k in ds})
