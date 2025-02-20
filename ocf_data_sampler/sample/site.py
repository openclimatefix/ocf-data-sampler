"""PVNet Site sample implementation for netCDF data handling and conversion"""

import xarray as xr

from typing import override

from ocf_data_sampler.sample.base import SampleBase, NumpySample
from ocf_data_sampler.torch_datasets.datasets.site import convert_netcdf_to_numpy_sample


class SiteSample(SampleBase):
    """Handles PVNet site specific netCDF operations"""

    def __init__(self, data: xr.Dataset):
        
        if not isinstance(data, xr.Dataset):
            raise TypeError(f"Data must be xarray Dataset - Found type {type(data)}")
        
        self._data = data

    @override
    def to_numpy(self) -> NumpySample:                    
        return convert_netcdf_to_numpy_sample(self._data)

    def save(self, path: str) -> None:
        """Save site sample data as netCDF
        
        Args:
            path: Path to save the netCDF file
        """
        self._data.to_netcdf(path, mode="w", engine="h5netcdf")

    @classmethod
    def load(cls, path: str) -> 'SiteSample':
        """Load site sample data from netCDF
        
        Args:
            path: Path to load the netCDF file from
        """
        return cls(xr.open_dataset(path))

    # TODO - placeholder for now
    def plot(self) -> None:
        raise NotImplementedError("Plotting not yet implemented for SiteSample")
