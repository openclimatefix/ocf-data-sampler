"""A lightweight DataArray-like class."""

from typing import Any

import numpy as np
import xarray as xr
from tensorstore import Future as TensorStoreFuture
from tensorstore import TensorStore
from xarray_tensorstore import _TensorStoreAdapter


class LightDataArray:
    """A lightweight DataArray-like class."""

    __slots__ = ["attrs", "coord_dims", "coords", "data", "dims", "future"]

    def __init__(
        self,
        data: np.ndarray | TensorStore,
        dims: tuple[str, ...],
        coords: dict[str, np.ndarray],
        coord_dims: dict[str, tuple[str, ...]],
        attrs: None | dict = None,
    ) -> None:
        """A lightweight DataArray-like class."""
        self.data = data
        self.dims = dims
        self.coords = coords
        self.coord_dims = coord_dims
        self.attrs = attrs or {}
        self.future: None | TensorStoreFuture = None

    @classmethod
    def from_xarray(cls, da: xr.DataArray) -> "LightDataArray":
        """Create a LightDataArray from an Xarray DataArray."""
        # Get raw data handle which can be a numpy array or TensorStore
        data: TensorStore | np.ndarray
        if isinstance(da.variable._data, _TensorStoreAdapter):
            data = da.variable._data.array
        elif isinstance(da.variable._data, np.ndarray):
            data = da.variable._data
        else:
            raise ValueError(f"Data backend of type {type(da.variable._data)} not supported.")

        coord_values: dict[str, np.ndarray] = {}
        coord_dims: dict[str, tuple[str, ...]] = {}

        for k, v in da.coords.items():
            if v.ndim <= 1:
                coord_values[k] = v.values
                coord_dims[k] = v.dims
            else:
                raise ValueError(
                    "Coordinates with more than 1 dimension not supported. "
                    f"Found coord '{k}' with shape {v.shape}.",
                )

        return cls(
            data=data,
            dims=da.dims,
            coords=coord_values,
            coord_dims=coord_dims,
            attrs=da.attrs,
        )

    def to_xarray(self) -> xr.DataArray:
        """Convert to an Xarray DataArray.

        Note this loads the data eagerly.
        """
        coords_dict = {}
        for c, v in self.coords.items():
            cdims = self.coord_dims.get(c, ())

            # If it's a 1D array and the dimension is still in our dims list
            if np.ndim(v) == 1 and cdims[0] in self.dims:
                coords_dict[c] = (cdims, v)
            else:
                # It's a scalar or a non-indexed coordinate
                coords_dict[c] = v

        return xr.DataArray(
            data=self.values,
            dims=self.dims,
            coords=coords_dict,
            attrs=self.attrs,
        )

    def isel(
        self,
        indexers: None | dict[str, int | slice | list] = None,
        **indexers_kwargs: object,
    ) -> "LightDataArray":
        """Select data by integer index along specified dimensions.

        Args:
            indexers: A dict with keys matching dimensions and values given by integers, slice
                objects or arrays. `indexer` can be an integer, slice or array-like.
            **indexers_kwargs: The keyword arguments form of indexers.
        """
        if indexers is not None:
            indexers_kwargs.update(indexers)

        axis_indexers = [slice(None)] * len(self.dims)
        new_coords = self.coords.copy()
        dims_to_remove = []

        for dim, indexer in indexers_kwargs.items():
            if dim not in self.dims:
                raise KeyError(
                    f"'{dim}' is not a valid dimension or coordinate for data with dimensions"
                    f"{self.dims}",
                )

            axis_indexers[self.dims.index(dim)] = indexer

            # Slice the coords which depend on this dimension
            for c_name, c_dim_name in self.coord_dims.items():
                if c_dim_name == (dim,):
                    new_coords[c_name] = new_coords[c_name][indexer]

            # Check if this dimension is being collapsed (e.g. an integer index like .isel(time=0))
            if isinstance(indexer, int | np.integer):
                dims_to_remove.append(dim)

        # Slice the underlying dta
        sliced_data = self.data[tuple(axis_indexers)]

        # Remove dims that have been reduced to points
        remaining_dims = tuple(d for d in self.dims if d not in dims_to_remove)

        # Remove dims from coords that have been reduced to points
        new_coord_dims = self.coord_dims.copy()
        for dim in dims_to_remove:
            for c_name, c_dim_name in self.coord_dims.items():
                if c_dim_name == (dim,):
                    new_coord_dims[c_name] = ()

        return LightDataArray(
            data=sliced_data,
            dims=remaining_dims,
            coords=new_coords,
            coord_dims=new_coord_dims,
            attrs=self.attrs,
        )

    def _to_index(self, dim: str, label: object) -> slice | int:
        coord = self.coords[dim]
        if isinstance(label, slice):
            # start: find first index >= label.start
            start = None
            if label.start is not None:
                start = np.searchsorted(coord, label.start, side="left")

            # stop: find first index > label.stop to ensure slice includes endpoints
            stop = None
            if label.stop is not None:
                stop = np.searchsorted(coord, label.stop, side="right")

            return slice(start, stop)
        else:
            return np.searchsorted(coord, label, side="left")

    def sel(
        self,
        indexers: None | dict[str, Any | slice | list] = None,
        **indexers_kwargs: object,
    ) -> "LightDataArray":
        """Select data by coordinate labels, converting them to indices.

        Args:
            indexers: A dict with keys matching dimensions and values given by scalars, slices or
                arrays of tick labels. For dimensions with multi-index, the indexer may also be a
                dict-like object with keys matching index level names.
            **indexers_kwargs: The keyword arguments form of indexers.
        """
        if indexers is not None:
            indexers_kwargs.update(indexers)

        isel_kwargs = {dim: self._to_index(dim, val) for dim, val in indexers_kwargs.items()}
        return self.isel(**isel_kwargs)

    def read(self) -> None:
        """Trigger reading of the data if it's a lazy handle."""
        if isinstance(self.data, TensorStore):
            self.future = self.data.read()

    def load(self) -> "LightDataArray":
        """Load data in-place and return self."""
        self.data = self.values
        self.future = None
        return self

    @property
    def values(self) -> np.ndarray:
        """Get the underlying data as numpy array, loading it if necessary."""
        if isinstance(self.data, TensorStore):
            # If TensorStore handle reading
            if self.future is None:
                return np.asarray(self.data.read().result())
            else:
                return np.asarray(self.future.result())
        else:
            return np.asarray(self.data)


    def __getattr__(self, name: str) -> "LightDataArray":
        """Allow access to coordinates via attribute syntax, e.g., da.time."""
        if name in self.coords:
            return self[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> "LightDataArray":
        """Allow access to coordinates via indexing syntax, e.g., da['time']."""
        if key in self.coords:
            return LightDataArray(
                data=self.coords[key],
                dims=self.coord_dims[key],
                coords={key: self.coords[key]},
                coord_dims={key: self.coord_dims[key]},
            )
        raise KeyError(f"Coordinate '{key}' not found.")

    def __getstate__(self) -> dict:
        """Prepare state for pickling, excluding un-picklable attributes."""
        return {
            "data": self.data,
            "dims": self.dims,
            "coords": self.coords,
            "attrs": self.attrs,
            "coord_dims": self.coord_dims,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        for k, v in state.items():
            setattr(self, k, v)
        # Restore the un-picklable attribute to a default state
        self.future = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the underlying data array."""
        return self.data.shape

    def __len__(self) -> int:
        """Return the length of the underlying data array."""
        return self.shape[0]
