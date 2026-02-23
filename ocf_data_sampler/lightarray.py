"""A lightweight DataArray-like class."""

from typing import Any

import numpy as np
import xarray as xr


class LightDataArray:
    """A lightweight DataArray-like class."""

    __slots__ = ["attrs", "coord_dims", "coords", "data", "dims", "future"]

    def __init__(
        self,
        data: np.ndarray,
        dims: tuple[str],
        coords: dict[str, np.ndarray],
        coord_dims: dict[str, str],
        attrs: None | dict = None,
    ) -> None:
        """A lightweight DataArray-like class."""
        self.data = data
        self.dims = dims
        self.coords = coords
        self.coord_dims = coord_dims
        self.attrs = attrs or {}
        self.future = None

    @classmethod
    def from_xarray(cls, da: xr.DataArray) -> "LightDataArray":
        """Create a LightDataArray from an Xarray DataArray."""
        # Get raw data handle
        raw_handle = da.variable._data
        while hasattr(raw_handle, "array"):
            raw_handle = raw_handle.array

        extracted_coords = {}
        coord_dims = {}

        for k, v in da.coords.items():
            # Only pull 1D coords to keep it fast
            if v.ndim == 1:
                extracted_coords[k] = v.values
                # Record which dimension this coordinate belongs to
                if hasattr(v, "dims") and len(v.dims) == 1:
                    coord_dims[k] = v.dims[0]
            elif v.ndim == 0:
                 extracted_coords[k] = v.values

        return cls(
            data=raw_handle,
            dims=da.dims,
            coords=extracted_coords,
            coord_dims=coord_dims,
            attrs=da.attrs,
        )

    def to_xarray(self) -> xr.DataArray:
        """Convert to an Xarray DataArray."""
        coords_dict = {}
        for c, v in self.coords.items():
            # Get the dimension name for this coordinate
            dim_name = self.coord_dims.get(c)

            # If it's a 1D array and the dimension is still in our dims list
            if dim_name in self.dims and getattr(v, "ndim", 0) > 0:
                coords_dict[c] = ([dim_name], v)
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

        indexer = [slice(None)] * len(self.dims)
        new_coords = self.coords.copy()
        dims_to_remove = []

        for dim, val in indexers_kwargs.items():
            if dim not in self.dims:
                continue

            axis = self.dims.index(dim)
            indexer[axis] = val

            # Find coordinates that depend on this dimension and slice them
            for c_name, c_dim_name in self.coord_dims.items():
                if c_dim_name == dim and c_name in new_coords:
                    new_coords[c_name] = new_coords[c_name][val]

            # Check if this dimension is being collapsed (integer index)
            if not isinstance(val, (slice, list, tuple, np.ndarray)):
                dims_to_remove.append(dim)

        sliced_data = self.data[tuple(indexer)]

        # Return raw value if it's a scalar
        if not hasattr(sliced_data, "ndim") or sliced_data.ndim == 0:
            return sliced_data

        # Update dims mapping for the new object
        remaining_dims = tuple(d for d in self.dims if d not in dims_to_remove)

        return LightDataArray(
            data=sliced_data,
            dims=remaining_dims,
            coords=new_coords,
            coord_dims=self.coord_dims,
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
        if hasattr(self.data, "read"):
            self.future = self.data.read()

    def load(self) -> "LightDataArray":
        """Load the data if it's not already a numpy array, and return self for chaining."""
        if isinstance(self.data, np.ndarray):
            return self

        if self.future is not None:
            self.data = np.asarray(self.future.result())
            self.future = None
        elif hasattr(self.data, "read"):
            self.data = np.asarray(self.data.read().result())
        else:
            self.data = np.asarray(self.data)
        return self

    @property
    def values(self) -> np.ndarray:
        """Get the underlying data as numpy array, loading it if necessary."""
        return self.load().data

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

    def __len__(self) -> int:
        """Return the length of the underlying data array."""
        return len(self.data)
