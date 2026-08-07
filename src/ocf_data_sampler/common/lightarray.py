"""A lightweight DataArray-like class."""

from typing import Any, TypedDict, overload

import numpy as np
import tensorstore as ts
import xarray as xr
from xarray_tensorstore import _TensorStoreAdapter

from ocf_data_sampler.common.types import Indexer


class LightDataArrayState(TypedDict):
    """Serialized state of a LightDataArray."""

    _data: np.ndarray | ts.TensorStore
    dims: tuple[str, ...]
    coords: dict[str, tuple[tuple[str, ...], np.ndarray]]
    attrs: dict[Any, Any]


def _validate_coords(
    data: np.ndarray | ts.TensorStore,
    dims: tuple[str, ...],
    coords: dict[str, tuple[tuple[str, ...], np.ndarray]]
) -> None:
    """Helper to validate the coordinates dictionary."""
    for coord_name, (coord_dims, coord_values) in coords.items():

        # Check types in coords
        if not isinstance(coord_name, str):
            raise TypeError(f"Coordinate name must be a string. Got {type(coord_name)}.")
        if not isinstance(coord_dims, tuple) or not all(isinstance(dim, str) for dim in coord_dims):
            raise TypeError(
                f"Coordinate dimensions must be a tuple of strings. Got {coord_dims}.",
            )
        if not isinstance(coord_values, np.ndarray):
            raise TypeError(
                f"Coordinate values must be a numpy array. Got {type(coord_values)}.",
            )

        # Check that the number of dimensions in the coordinate values matches the number of
        # dimensions
        if not len(coord_dims) == coord_values.ndim:
            raise ValueError(
                f"Number of dimensions for coordinate '{coord_name}' ({len(coord_dims)}) must "
                f"match the number of dimensions in its values ({coord_values.ndim}).",
            )

        # We allow non-dimensional coordinates (i.e. coords not in dims) to be multidimensional, so
        # check that all the dimensions in coord_dims are present in coords
        for dim in coord_dims:
            if dim not in coords:
                raise ValueError(
                    f"Dimension '{dim}' for coordinate '{coord_name}' is not present in the "
                     "coordinates dictionary.",
                )

    for dim in dims:
        # Check types
        if not isinstance(dim, str):
            raise TypeError(f"Dimension names must be strings. Got {type(dim)}.")
        # Check that all dimensions are present in coords
        if dim not in coords:
            raise ValueError(f"Dimension '{dim}' is not present in the coordinates dictionary.")

    # Check that the number of dimensions in dims matches the number of dimensions in the data
    if not len(dims) == data.ndim:
        raise ValueError(
            f"Number of dimensions ({len(dims)}) must match the number of dimensions "
            f"({data.ndim}) in the data.",
        )

    for i, dim in enumerate(dims):
        if data.shape[i] != coords[dim][1].shape[0]:
            raise ValueError(
                f"Dimension '{dim}' has size {data.shape[i]} in the data but size "
                f"{coords[dim][1].shape[0]} in the coordinates.",
            )


def _sanitise_indexer(dim: str, dim_size: int, indexer: Indexer) -> Indexer:
    """Normalise an indexer so dimension-collapse logic and indexing agree on its semantics."""
    if isinstance(indexer, bool | np.bool_):
        raise TypeError(
            f"Found scalar boolean indexer on dimension '{dim}'. Scalar boolean indexers are not "
            "supported."
        )

    # Pass the expected types through unchanged
    if isinstance(indexer, int | np.integer | slice):
        return indexer

    # Convert array-likes (e.g. list, tuple) to numpy arrays
    indexer = np.asarray(indexer)

    # Convert 0-d integer arrays to plain ints
    if indexer.ndim == 0:
        if not np.issubdtype(indexer.dtype, np.integer):
            raise TypeError(
                f"0-d indexer for dimension '{dim}' must be of integer type. "
                f"Got dtype {indexer.dtype}.",
            )
        return int(indexer)

    # Check that the indexer is at most 1-dimensional
    if indexer.ndim > 1:
        raise ValueError(
            f"Indexer for dimension '{dim}' must be at most 1-dimensional. "
            f"Got {indexer.ndim} dimensions.",
        )

    if np.issubdtype(indexer.dtype, np.bool_):
        if len(indexer) != dim_size:
            raise IndexError(
                f"Boolean indexer for dimension '{dim}' has length {len(indexer)}, but the "
                f"dimension has length {dim_size}.",
            )
        return indexer

    if not np.issubdtype(indexer.dtype, np.integer):
        raise TypeError(
            f"Indexer for dimension '{dim}' must have an integer or boolean dtype. "
            f"Got dtype {indexer.dtype}.",
        )

    return indexer


def _normalise_index_origin(data: np.ndarray | ts.TensorStore) -> np.ndarray | ts.TensorStore:
    """Translate TensorStore domains to zero so subsequent indexing remains positional."""
    if isinstance(data, ts.TensorStore):
        return data[ts.d[:].translate_to[0]]
    return np.asarray(data)


@overload
def _apply_axis_indexers(
    data: np.ndarray,
    axis_indexers: tuple[Indexer, ...],
) -> np.ndarray: ...


@overload
def _apply_axis_indexers(
    data: ts.TensorStore,
    axis_indexers: tuple[Indexer, ...],
) -> ts.TensorStore: ...


def _apply_axis_indexers(
    data: np.ndarray | ts.TensorStore,
    axis_indexers: tuple[Indexer, ...],
) -> np.ndarray | ts.TensorStore:

    if len(axis_indexers) != data.ndim:
        raise ValueError(
            f"Number of indexers ({len(axis_indexers)}) must match the number of dimensions "
            f"({data.ndim}) in the data.",
        )

    # If all indexers are integers or slices, we can apply them directly to the data in one step.
    # This is the fastest path and is the most common case, so we check for it first.
    if all(isinstance(indexer, int | np.integer | slice) for indexer in axis_indexers):
        return _normalise_index_origin(data[axis_indexers])

    # Apply basic indexers together, then array indexers one axis at a time to preserve xarray's
    # orthogonal indexing semantics without slowing down the common int-and-slice path.
    basic_indexers = tuple(
        indexer if isinstance(indexer, int | np.integer | slice) else slice(None)
        for indexer in axis_indexers
    )
    sliced_data = _normalise_index_origin(data[basic_indexers])

    current_axis = 0
    for indexer in axis_indexers:

        # Slices are already applied in the basic_indexers step, so we skip them here and
        # increment the current_axis counter to keep track of which axis we're on in the sliced data
        if isinstance(indexer, slice):
            current_axis += 1
            continue

        # Dimensions that have been reduced to points do not have an axis in the sliced data, so
        # we don't increment current_axis for these dimensions
        elif isinstance(indexer, int | np.integer):
            continue

        elif isinstance(indexer, np.ndarray | list | tuple):
            axis_indexer = tuple(
                indexer if i == current_axis else slice(None) for i in range(current_axis + 1)
            )
            sliced_data = _normalise_index_origin(sliced_data[axis_indexer])
            current_axis += 1

        else:
            raise TypeError(
                f"Unsupported indexer type: {type(indexer)}. Must be int, slice, or array-like.",
            )

    return sliced_data



class LightDataArray:
    """A lightweight DataArray-like class."""

    __slots__ = ["_data", "_future", "attrs", "coords", "dims"]

    def __init__(
        self,
        data: np.ndarray | ts.TensorStore,
        coords: dict[str, tuple[tuple[str, ...], np.ndarray]],
        dims: tuple[str, ...],
        attrs: dict[Any, Any] | None = None,
        skip_validation: bool = False,
    ) -> None:
        """A lightweight DataArray-like class.

        Args:
            data: Values for this array
            coords: Coordinates (tick labels) to use for indexing along each dimension. Must be of
                the form {coord name: (tuple of dimension names, array-like)}.
            dims: The dimension names corresponding to the axes of the data
            attrs: Attributes to assign to the new instance
            skip_validation: Skip validation of the coords against the data and dims. Intended
                for internal use where inputs are constructed to be consistent.
        """
        if not skip_validation:
            _validate_coords(data, dims, coords)

        self._data = data
        self.coords = coords
        self.dims = dims
        self.attrs = attrs or {}
        self._future: ts.Future[Any] | None = None

    @property
    def data(self) -> np.ndarray | ts.TensorStore:
        """Return the backing array."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray | ts.TensorStore) -> None:
        """Replace the backing array without changing dimensions."""
        if value.shape != self.shape:
            raise ValueError(
                "Replacement data must match the existing shape. "
                f"Replacement has shape {value.shape}; existing data has shape {self.shape}.",
            )

        self._data = value
        self._future = None

    @classmethod
    def from_xarray(cls, da: xr.DataArray) -> "LightDataArray":
        """Create a LightDataArray from an Xarray DataArray."""
        # Get raw data handle which can be a numpy array or TensorStore
        data: ts.TensorStore | np.ndarray
        if isinstance(da.variable._data, _TensorStoreAdapter):
            data = da.variable._data.array
        elif isinstance(da.variable._data, np.ndarray):
            data = da.variable._data
        else:
            raise ValueError(f"Data backend of type {type(da.variable._data)} not supported.")

        dims = tuple(str(d) for d in da.dims)
        coords = {
            str(k): (tuple(str(dim) for dim in v.dims), v.values) for k, v in da.coords.items()
        }

        return cls(
            data=data,
            dims=dims,
            coords=coords,
            attrs=da.attrs,
            skip_validation=True,
        )

    def to_xarray(self) -> xr.DataArray:
        """Convert to an Xarray DataArray."""
        return xr.DataArray(
            data=self.data,
            dims=self.dims,
            coords=self.coords,
            attrs=dict(self.attrs),
        )

    def isel(
        self,
        indexers: dict[str, Indexer] | None = None,
        **indexers_kwargs: Indexer,
    ) -> "LightDataArray":
        """Select data by integer index along specified dimensions.

        Args:
            indexers: A dict with keys matching dimensions and values given by integers, slice
                objects or arrays. `indexer` can be an integer, slice or array-like.
            **indexers_kwargs: The keyword arguments form of indexers.

        Returns:
            A new LightDataArray. Attributes are copied, while coordinate arrays may be views of
            the original coordinate arrays rather than independent copies.
        """
        if indexers is not None:
            indexers_kwargs.update(indexers)

        # Validate that all indexers correspond to valid dimensions
        for dim in indexers_kwargs:
            if dim not in self.dims:
                raise KeyError(
                    f"'{dim}' is not a valid dimension or coordinate for data with dimensions"
                    f"{self.dims}",
                )

        # Normalise indexers to simplify the logic for applying them to the data and coordinates
        indexers_kwargs = {
            dim: _sanitise_indexer(dim, self.shape[self.dims.index(dim)], indexer)
            for dim, indexer in indexers_kwargs.items()
        }

        # Slice the data
        axis_indexers = tuple(indexers_kwargs.get(dim, slice(None)) for dim in self.dims)
        sliced_data = _apply_axis_indexers(self.data, axis_indexers)

        # Remove dimensions that have been reduced to single points (i.e., indexed by an integer)
        new_dims = tuple(
            dim for dim in self.dims if not isinstance(indexers_kwargs.get(dim), int | np.integer)
        )

        # Slice the coordinate values
        indexed_dims = indexers_kwargs.keys()
        new_coords = self.coords.copy()
        for coord_name, (coord_dims, coord_values) in new_coords.items():
            # If the coordinate is not indexed along any of its dimensions, copy it as is, else
            # slice the coordinate values along the indexed dimensions
            if indexed_dims.isdisjoint(coord_dims):
                new_coords[coord_name] = (coord_dims, coord_values)
            else:
                coord_axis_indexers = tuple(
                    indexers_kwargs.get(dim, slice(None)) for dim in coord_dims
                )
                sliced_coord_values = _apply_axis_indexers(coord_values, coord_axis_indexers)
                new_coord_dims = tuple(dim for dim in coord_dims if dim in new_dims)
                new_coords[coord_name] = (new_coord_dims, sliced_coord_values)

        return LightDataArray(
            data=sliced_data,
            dims=new_dims,
            coords=new_coords,
            attrs=dict(self.attrs),
            skip_validation=True,
        )

    def read(self) -> None:
        """Trigger reading of the data if it's a lazy handle."""
        if isinstance(self.data, ts.TensorStore):
            self._future = self.data.read()

    def load(self) -> "LightDataArray":
        """Load data in-place and return self."""
        self.data = self.values
        return self

    @property
    def values(self) -> np.ndarray:
        """Return the data as a NumPy array, loading it if necessary."""
        data = self._data

        if not isinstance(data, ts.TensorStore):
            return np.asarray(data)

        if self._future is None:
            return np.asarray(data.read().result())

        return np.asarray(self._future.result())

    def __getitem__(self, key: str) -> "LightDataArray":
        """Allow access to coordinates via indexing syntax, e.g., da['time']."""
        if key in self.coords:
            coord_dims, coord_values = self.coords[key]
            return LightDataArray(
                data=coord_values,
                dims=coord_dims,
                coords={key: (coord_dims, coord_values)},
                skip_validation=True,
            )
        raise KeyError(f"Coordinate '{key}' not found.")

    def __getstate__(self) -> LightDataArrayState:
        """Prepare state for pickling, excluding un-picklable attributes."""
        return {
            "_data": self._data,
            "dims": self.dims,
            "coords": self.coords,
            "attrs": self.attrs,
        }

    def __setstate__(self, state: LightDataArrayState) -> None:
        """Restore state after unpickling."""
        self._data = state["_data"]
        self.dims = state["dims"]
        self.coords = state["coords"]
        self.attrs = state["attrs"]
        # Restore the un-picklable attribute to a default state
        self._future = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data array."""
        return self.data.shape

    def __len__(self) -> int:
        """Return the length of the underlying data array."""
        return self.shape[0]
