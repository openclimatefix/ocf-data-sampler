"""Module for opening NWP data."""


from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ProviderSpec:
    """Spatial coordinate names and optional validation for an NWP provider."""

    x_coord: str
    y_coord: str
    validator: Callable[[Sequence[xr.Dataset]], None] | None = None


# Provider-specific coordinates and validation share one loading path.
PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "ukv": ProviderSpec("x_osgb", "y_osgb"),
    "ecmwf": ProviderSpec("longitude", "latitude"),
    "mo_global": ProviderSpec("longitude", "latitude"),
    "gencast": ProviderSpec("longitude", "latitude"),
    "fgn": ProviderSpec("longitude", "latitude"),
    "cloudcasting": ProviderSpec(
        "x_geostationary", "y_geostationary", validate_matching_projections,
    ),
}


def open_nwp(zarr_path: ZarrSource, provider: str) -> xr.DataArray:
    """Open NWP data and conform it to the standard layout."""
    provider = provider.lower()

    if provider not in PROVIDER_REGISTRY:
        supported = ", ".join(sorted(PROVIDER_REGISTRY))
        raise ValueError(f"Unknown provider: {provider!r}. Supported: {supported}")

    spec = PROVIDER_REGISTRY[provider]

    ds = open_zarr_paths(zarr_path, concat_dim="init_time_utc", validator=spec.validator)
    ds = _rename(ds, name_mapping={"variable": "channel"})

    expected_coord_dtypes = {
        "init_time_utc": np.datetime64,
        "step": np.timedelta64,
        "channel": np.str_,
        spec.x_coord: np.number,
        spec.y_coord: np.number,
    }
    validate_coords(
        ds,
        expected_coord_dtypes,
        source=f"NWP provider {provider!r}",
    )

    ds = make_spatial_coords_increasing(ds, x_coord=spec.x_coord, y_coord=spec.y_coord)

    assert_values_unique_increasing(ds["init_time_utc"].values, "init_time_utc")
    assert_values_unique_increasing(ds["step"].values, "step")
    _assert_steps_uniformly_spaced(ds["step"].values, provider)

    da = extract_only_data_array(ds)
    da = da.transpose("init_time_utc", "step", "channel", spec.x_coord, spec.y_coord)

    if not np.issubdtype(da.dtype, np.floating):
        raise TypeError(f"NWP data for {provider} should be floating, not {da.dtype}")

    return da


def _assert_steps_uniformly_spaced(steps: NDArray[np.timedelta64], provider: str) -> None:
    """Assert that the forecast steps are evenly spaced.

    Args:
        steps: The forecast steps, already checked to be unique and increasing.
        provider: The NWP provider name, used in the error message.
    """
    spacings = np.unique(np.diff(steps))

    if len(spacings) > 1:
        raise ValueError(
            f"NWP steps for provider {provider!r} must be evenly spaced, but the steps are "
            f"spaced by {spacings}. Mixed step spacing is not supported.",
        )


def _rename(ds: xr.Dataset, name_mapping: Mapping[str, str]) -> xr.Dataset:
    """Renames coordinates in the dataset based on the provided mapping.

    Args:
        ds: The xarray.Dataset to rename.
        name_mapping: A dictionary mapping old variable names to new names.
    """
    # Only rename variables that are actually present in the dataset
    return ds.rename({k: v for k, v in name_mapping.items() if k in ds})
