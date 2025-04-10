"""PVNet Site sample implementation for netCDF data handling and conversion."""

import xarray as xr
from typing_extensions import override

from ocf_data_sampler.numpy_sample.common_types import NumpySample
from ocf_data_sampler.torch_datasets.datasets.site import convert_netcdf_to_numpy_sample

from .base import SampleBase


class SiteSample(SampleBase):
    """Handles PVNet site specific netCDF operations."""

    def __init__(self, data: xr.Dataset) -> None:
        """Initializes the SiteSample object with the given xarray Dataset."""
        if not isinstance(data, xr.Dataset):
            raise TypeError(f"Data must be xarray Dataset - Found type {type(data)}")
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:
        return convert_netcdf_to_numpy_sample(self._data)

    @override
    def save(self, path: str) -> None:
        # Saves as NetCDF
        self._data.to_netcdf(path, mode="w", engine="h5netcdf")

    @classmethod
    @override
    def load(cls, path: str) -> "SiteSample":
        # Loads from NetCDF
        return cls(xr.open_dataset(path))

    @override
    def plot(self) -> None:
        # TODO - placeholder for now
        raise NotImplementedError("Plotting not yet implemented for SiteSample")
