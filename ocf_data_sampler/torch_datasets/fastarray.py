import numpy as np
import xarray as xr
from typing import Any


class FastDataArray:
    __slots__ = ['data', 'dims', 'coords', 'attrs', 'coord_dims', 'future']

    def __init__(
        self, 
        data: np.ndarray, 
        dims: tuple[str], 
        coords: dict[str, np.ndarray], 
        coord_dims: dict[str, str], 
        attrs: None | dict = None
    ):
        self.data = data
        self.dims = dims
        self.coords = coords
        self.coord_dims = coord_dims
        self.attrs = attrs or {}
        self.future = None

    @classmethod
    def from_xarray(cls, da: xr.DataArray) -> "FastDataArray":
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
                if hasattr(v, 'dims') and len(v.dims) == 1:
                    coord_dims[k] = v.dims[0]
            elif v.ndim == 0:
                 extracted_coords[k] = v.values

        return cls(
            data=raw_handle,
            dims=da.dims,
            coords=extracted_coords,
            coord_dims=coord_dims,
            attrs=da.attrs
        )

    def to_xarray(self) -> xr.DataArray:
        coords_dict = {}
        for c, v in self.coords.items():
            # Get the dimension name for this coordinate
            dim_name = self.coord_dims.get(c)
            
            # If it's a 1D array and the dimension is still in our dims list
            if dim_name in self.dims and getattr(v, 'ndim', 0) > 0:
                coords_dict[c] = ([dim_name], v)
            else:
                # It's a scalar or a non-indexed coordinate
                coords_dict[c] = v

        return xr.DataArray(
            data=self.values,
            dims=self.dims,
            coords=coords_dict,
            attrs=self.attrs
        )
    
    def isel(self, indexers=None, **kwargs) -> "FastDataArray":
        if indexers is not None:
            kwargs.update(indexers)

        indexer = [slice(None)] * len(self.dims)
        new_coords = self.coords.copy()
        dims_to_remove = []

        for dim, val in kwargs.items():
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

        return FastDataArray(
            data=sliced_data,
            dims=remaining_dims,
            coords=new_coords,
            coord_dims=self.coord_dims,
            attrs=self.attrs
        )

    def _to_index(self, dim: str, label: slice | Any) -> slice | int:
        coord = self.coords[dim]
        if isinstance(label, slice):
            # start: find first index >= label.start
            start = None
            if label.start is not None:
                start = np.searchsorted(coord, label.start, side='left')
            
            # stop: find first index > label.stop to ensure slice includes endpoints
            stop = None
            if label.stop is not None:
                stop = np.searchsorted(coord, label.stop, side='right')
                
            return slice(start, stop)
        else:
            return np.searchsorted(coord, label, side='left')

    def sel(self, indexers=None, **kwargs) -> "FastDataArray":
        if indexers is not None:
            kwargs.update(indexers)
        isel_kwargs = {dim: self._to_index(dim, val) for dim, val in kwargs.items()}
        return self.isel(**isel_kwargs)
    
    def read(self) -> None:
        if hasattr(self.data, "read"):
            self.future = self.data.read()

    def load(self) -> "FastDataArray":
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
        return self.load().data

    def __getattr__(self, name):
        if name in self.coords:
            return self.coords[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        if key in self.coords:
            return self.coords[key]
        raise KeyError(f"Coordinate '{key}' not found.")
    
    def __getstate__(self) -> dict:
        """
        Prepare state for pickling. 
        We must exclude 'future' because async objects cannot be pickled.
        """
        return {
            'data': self.data,
            'dims': self.dims,
            'coords': self.coords,
            'attrs': self.attrs,
            'coord_dims': self.coord_dims,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling"""
        for k, v in state.items():
            setattr(self, k, v)
        # Restore the un-picklable attribute to a default state
        self.future = None