"""Utilities for opening and lazily concatenating TensorStore-backed Xarray datasets.

Note: this module relies on `xarray_tensorstore` internals (`_TensorStoreAdapter`) to reach
the underlying TensorStore without materialising data. The dependency is pinned in
pyproject.toml; upgrades need checking against this.
"""

from collections.abc import Callable, Sequence
from glob import glob, has_magic

import tensorstore as ts
import xarray as xr
import xarray_tensorstore as xrt

from ocf_data_sampler.common.types import ZarrSource


def read(da: xr.DataArray) -> None:
    """Start a TensorStore read in-place without waiting; leave other backends unchanged.

    Replaces the adapter on the existing Variable, so arrays sharing that Variable see the change.
    """
    data = da.variable._data
    if isinstance(data, xrt._TensorStoreAdapter):
        da.data = data.read()


def _extract_tensorstore(da: xr.DataArray) -> ts.TensorStore:
    """Extract the backing TensorStore, or fail with a message that says why."""
    data = da.variable._data
    if not isinstance(data, xrt._TensorStoreAdapter):
        raise TypeError(f"{da.name!r} is backed by {type(data).__name__}, expected TensorStore.")
    return data.array


def _validate_concat_compatibility(datasets: Sequence[xr.Dataset], concat_dim: str) -> None:
    first, *rest = datasets
    if concat_dim not in first.dims:
        raise ValueError(f"{concat_dim!r} is not a dimension: {tuple(first.dims)}")

    for i, ds in enumerate(rest, start=1):

        # All coords and data_vars must be present in all datasets
        if set(ds.coords) != set(first.coords):
            raise ValueError(
                f"dataset {i}: coords {sorted(ds.coords, key=str)} "
                f"!= {sorted(first.coords, key=str)}"
            )
        if set(ds.data_vars) != set(first.data_vars):
            raise ValueError(
                f"dataset {i}: data_vars {sorted(ds.data_vars, key=str)} "
                f"!= {sorted(first.data_vars, key=str)}"
            )

        # All dims except concat_dim must match in size
        for dim, size in first.sizes.items():
            if dim != concat_dim and ds.sizes.get(dim) != size:
                raise ValueError(f"dataset {i}: {dim}={ds.sizes.get(dim)}, expected {size}")

        # All data_vars must have the same dims and dtype
        for name in first.data_vars:
            if ds[name].dims != first[name].dims or ds[name].dtype != first[name].dtype:
                raise ValueError(
                    f"dataset {i}: {name!r} is {ds[name].dims}/{ds[name].dtype}, "
                    f"expected {first[name].dims}/{first[name].dtype}"
                )

        # All coords and data_vars which don't contain the concat_dim dimension must be identical
        # Note: `.equals()` reads lazy data into memory. This is fine for coords and static vars,
        # which should be small
        for name in [*first.coords, *first.data_vars]:
            if concat_dim not in first[name].dims and not ds[name].equals(first[name]):
                raise ValueError(
                    f"dataset {i}: {name!r} does not span {concat_dim!r} but differs"
                )


def concat_tensorstore(datasets: Sequence[xr.Dataset], concat_dim: str) -> xr.Dataset:
    """Concatenate tensorstore-backed Datasets along an existing dimension, lazily.

    Data variables containing the `concat_dim` dimension are concatenated lazily using TensorStore.
    Everything else must match across datasets and is taken from the first dataset, as are attrs.

    Args:
        datasets: Sequence of Datasets to concatenate.
        concat_dim: Dimension along which to concatenate.
    """
    datasets = list(datasets)
    if len(datasets) < 2:
        raise ValueError("need at least two datasets")
    _validate_concat_compatibility(datasets, concat_dim)
    first = datasets[0]

    # Create a new shell dataset which contains only the concatenated coords. We will handle the
    # data_vars separately so we can lazily concatenate them with tensorstore.
    # - combine_attrs="override" keeps the attrs of the first dataset, which is the behaviour we
    #   copy for the data_vars below.
    # - join="exact" ensures that the coords are identical across datasets, which is a bakstop for
    #   the _validate() check above.
    ds_out = xr.concat(
        [ds.drop_vars(first.data_vars) for ds in datasets],
        dim=concat_dim,
        join="exact",
        combine_attrs="override",
    )

    for name, da in first.data_vars.items():
        if concat_dim in da.dims:
            store = ts.concat(
                [_extract_tensorstore(ds[name]) for ds in datasets],
                axis=da.dims.index(concat_dim),
            )
            ds_out[name] = xr.Variable(da.dims, xrt._TensorStoreAdapter(store), attrs=da.attrs)
        else:
            ds_out[name] = da.variable  # attrs travel with the Variable; xarray copies on assign
    return ds_out



def resolve_zarr_paths(zarr_path: ZarrSource) -> list[str]:
    """Resolve a path, local glob, or sequence into a non-empty list of paths."""
    if isinstance(zarr_path, str):
        paths = sorted(glob(zarr_path)) if has_magic(zarr_path) else [zarr_path]
    else:
        paths = list(zarr_path)

    if not paths:
        raise ValueError(f"No Zarr stores found for {zarr_path!r}")

    return paths


def open_zarr_paths(
    zarr_path: ZarrSource,
    concat_dim: str | None = None,
    *,
    validator: Callable[[Sequence[xr.Dataset]], None] | None = None,
) -> xr.Dataset:
    """Open one or more Zarr stores using TensorStore.

    Args:
        zarr_path: A path, local glob pattern, or sequence of paths.
        concat_dim: Dimension along which multiple stores are concatenated.
        validator: Optional callback called with opened datasets before returning or concatenating.
    """
    paths = resolve_zarr_paths(zarr_path)

    datasets = [xrt.open_zarr(p) for p in paths]
    if validator is not None:
        validator(datasets)

    if len(datasets) == 1:
        return datasets[0]

    if concat_dim is None:
        raise ValueError("`concat_dim` must be specified when opening multiple Zarr stores")

    return concat_tensorstore(datasets, concat_dim)
