# uk_regional.py

""" 
PVNet - UK Regional sample / dataset 
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
from pathlib import Path

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict
from ocf_data_sampler.select import Location, slice_datasets_by_space, slice_datasets_by_time

from ocf_data_sampler.numpy_sample import (
    NWPSampleKey, 
    GSPSampleKey,
    SatelliteSampleKey,
    convert_nwp_to_numpy_sample,
    convert_gsp_to_numpy_sample,
    convert_satellite_to_numpy_sample,
    make_sun_position_numpy_sample
)

from ocf_data_sampler.torch_datasets.pvnet_uk_regional import get_gsp_locations
from ocf_data_sampler.utils import minutes

from ocf_data_sampler.sample.base import SampleBase


class PVNetSample(SampleBase):
    """ Sample class specific to PVNet """
    
    REQUIRED_KEYS = {
        'nwp',
        GSPSampleKey.gsp,
        SatelliteSampleKey.satellite_actual,
        GSPSampleKey.solar_azimuth,
        GSPSampleKey.solar_elevation
    }

    def __init__(self):
        super().__init__()

    def validate(self) -> None:
        # Check required keys
        missing_keys = self.REQUIRED_KEYS - set(self.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
            
        # Validate NWP structure
        if 'nwp' in self._data and not isinstance(self._data['nwp'], dict):
            raise TypeError("NWP data must be nested dictionary")

    def plot(self, **kwargs) -> None:

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        if GSPSampleKey.gsp in self._data:
            axes[0, 0].plot(self._data[GSPSampleKey.gsp])
            axes[0, 0].set_title('GSP Generation')
        
        if 'nwp' in self._data:
            first_nwp = list(self._data['nwp'].values())[0]
            if 'nwp' in first_nwp:
                axes[0, 1].imshow(first_nwp['nwp'][0])
                axes[0, 1].set_title('NWP (First Channel)')
        
        if SatelliteSampleKey.satellite_actual in self._data:
            axes[1, 0].imshow(self._data[SatelliteSampleKey.satellite_actual])
            axes[1, 0].set_title('Satellite Data')
        
        if GSPSampleKey.solar_azimuth in self._data and GSPSampleKey.solar_elevation in self._data:
            axes[1, 1].plot(self._data[GSPSampleKey.solar_azimuth], label='Azimuth')
            axes[1, 1].plot(self._data[GSPSampleKey.solar_elevation], label='Elevation')
            axes[1, 1].set_title('Solar Position')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


class PVNetUKRegionalDataset(Dataset):
    """ PVNet UK Regional """
    
    def __init__(
        self, 
        config_filename: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        gsp_ids: Optional[List[int]] = None,
    ):

        self.config = load_yaml_configuration(config_filename)
        self.datasets_dict = get_dataset_dict(self.config)
        
        # Get valid time periods
        from ocf_data_sampler.torch_datasets.valid_time_periods import find_valid_time_periods
        valid_time_periods = find_valid_time_periods(self.datasets_dict, self.config)
        
        # Convert to datetime index with proper frequency
        self.valid_t0_times = pd.date_range(
            valid_time_periods.index[0],
            valid_time_periods.index[-1],
            freq=minutes(self.config.input_data.gsp.time_resolution_minutes)
        )
        
        # Apply time filters
        if start_time is not None:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times >= pd.Timestamp(start_time)
            ]
        if end_time is not None:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times <= pd.Timestamp(end_time)
            ]
            
        # Get locations
        self.locations = get_gsp_locations(gsp_ids)
        self.location_lookup = {loc.id: loc for loc in self.locations}
        
        # Create sampling indices
        t_index, loc_index = np.meshgrid(
            np.arange(len(self.valid_t0_times)),
            np.arange(len(self.locations)),
        )
        self.index_pairs = np.stack((t_index.ravel(), loc_index.ravel())).T

    def __len__(self) -> int:
        return len(self.index_pairs)

    def _create_sample(self, t0: pd.Timestamp, location: Location) -> PVNetSample:

        sample = PVNetSample()
        
        # Get sliced data
        sample_dict = slice_datasets_by_space(self.datasets_dict, location, self.config)
        sample_dict = slice_datasets_by_time(sample_dict, t0, self.config)
        
        # Process NWP data
        if "nwp" in sample_dict:
            nwp_data = {}
            for nwp_key, da_nwp in sample_dict["nwp"].items():
                nwp_data[nwp_key] = convert_nwp_to_numpy_sample(da_nwp)
            sample['nwp'] = nwp_data
            
        # Process satellite data
        if "sat" in sample_dict:
            sample.update(convert_satellite_to_numpy_sample(sample_dict["sat"]))
            
        # Process GSP data
        if "gsp" in sample_dict:
            gsp_data = convert_gsp_to_numpy_sample(
                sample_dict["gsp"],
                t0_idx=-self.config.input_data.gsp.interval_start_minutes 
                        / self.config.input_data.gsp.time_resolution_minutes
            )
            sample.update(gsp_data)
            
        # Get sun position data
        gsp_config = self.config.input_data.gsp
        datetimes = pd.date_range(
            t0 + minutes(gsp_config.interval_start_minutes),
            t0 + minutes(gsp_config.interval_end_minutes),
            freq=minutes(gsp_config.time_resolution_minutes),
        )
        
        if location.coordinate_system == "osgb":
            from ocf_data_sampler.select.geospatial import osgb_to_lon_lat
            lon, lat = osgb_to_lon_lat(location.x, location.y)
        else:
            lon, lat = location.x, location.y
            
        sun_data = make_sun_position_numpy_sample(datetimes, lon, lat)
        sample.update(sun_data)
        
        # Fill any NaN values
        sample.fill_nans()
        
        return sample

    def __getitem__(self, idx: int) -> PVNetSample:

        t_index, loc_index = self.index_pairs[idx]
        location = self.locations[loc_index]
        t0 = self.valid_t0_times[t_index]
        
        return self._create_sample(t0, location)
    
    def get_sample(self, t0: pd.Timestamp, gsp_id: int) -> PVNetSample:

        if t0 not in self.valid_t0_times:
            raise ValueError(f"Invalid t0 time: {t0}")
        if gsp_id not in self.location_lookup:
            raise ValueError(f"Invalid GSP ID: {gsp_id}")
            
        location = self.location_lookup[gsp_id]
        return self._create_sample(t0, location)
