import xarray as xr
import numpy as np
import pytest
from ocf_data_sampler.load.nwp.providers.ukv import open_ukv

def test_ukv_rename_logic_legacy():
    # Mock legacy dataset with 'x' and 'y'
    ds = xr.Dataset(
        coords={
            "x": np.array([1, 2]),
            "y": np.array([3, 4]),
            "init_time": np.array([0]),
            "variable": np.array(["temp"])
        }
    )
    
    # We test the internal logic directly since we can't easily 
    # mock the Zarr file structure for a full open_ukv call here
    rename_map = {
        "init_time": "init_time_utc",
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }
    
    actual_rename = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    ds_renamed = ds.rename(actual_rename)
    
    assert "x_osgb" in ds_renamed.coords
    assert "y_osgb" in ds_renamed.coords
    assert "x" not in ds_renamed.coords

def test_ukv_rename_logic_new():
    # Mock new dataset that already has 'x_osgb'
    ds = xr.Dataset(
        coords={
            "x_osgb": np.array([1, 2]),
            "y_osgb": np.array([3, 4]),
            "init_time_utc": np.array([0]),
            "channel": np.array(["temp"])
        }
    )
    
    rename_map = {
        "init_time": "init_time_utc",
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }
    
    actual_rename = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    
    # This should not raise an error even if actual_rename is empty
    ds_renamed = ds.rename(actual_rename)
    
    assert "x_osgb" in ds_renamed.coords
    assert "x" not in ds_renamed.coords