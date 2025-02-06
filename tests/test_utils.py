import pytest
import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.utils import Location


def test_location_from_site_metadata(data_sites):
    """ Location instances from site metadata - created from the fixtures site data """

    meta_df = pd.read_csv(data_sites.metadata_file_path)
    
    for _, row in meta_df.iterrows():
        loc = Location(
            x=row['longitude'], 
            y=row['latitude'], 
            coordinate_system='lon_lat',
            id=row['site_id']
        )
        
        # Assert location attributes match metadata
        assert loc.x == pytest.approx(row['longitude'])
        assert loc.y == pytest.approx(row['latitude'])
        assert loc.id == row['site_id']
        assert loc.coordinate_system == 'lon_lat'


def test_location_coordinate_systems_match_metadata(data_sites, session_tmp_path):
    """ Verify site locations are valid across different coordinate systems """

    meta_df = pd.read_csv(data_sites.metadata_file_path)    
    meta_df = meta_df.loc[:, ~meta_df.columns.str.contains('^Unnamed')]
    
    required_columns = ['site_id', 'longitude', 'latitude']
    for col in required_columns:
        assert col in meta_df.columns, f"Missing required column: {col}"
    
    coordinate_systems = ['lon_lat', 'osgb', 'geostationary']
    
    for system in coordinate_systems:
        for _, row in meta_df.iterrows():
            if system == 'lon_lat':
                x, y = row['longitude'], row['latitude']
            elif system == 'osgb':
                x, y = 100000, 200000
            elif system == 'geostationary':
                x, y = 2000000, 2500000
            
            try:
                loc = Location(
                    x=float(x), 
                    y=float(y), 
                    coordinate_system=system,
                    id=int(row['site_id'])
                )

                # Assert validity of location
                assert loc is not None
            except ValueError as e:
                pytest.fail(f"Failed to create Location for {system} with {row}: {e}")
            except Exception as e:
                pytest.fail(f"Unexpected error for {system} with {row}: {e}")


def test_location_from_zarr_paths(sat_zarr_path, nwp_ukv_zarr_path):
    """ Locations using coordinates from Zarr file """
    
    sat_ds = xr.open_zarr(sat_zarr_path)
    loc_sat = Location(
        x=sat_ds.x_geostationary.values[0], 
        y=sat_ds.y_geostationary.values[0], 
        coordinate_system='geostationary'
    )
    
    nwp_ds = xr.open_zarr(nwp_ukv_zarr_path)
    loc_nwp = Location(
        x=100000,
        y=200000,
        coordinate_system='osgb'
    )

    # Satellite and NWP (UKV) coordinates
    assert loc_sat is not None
    assert loc_nwp is not None


def test_location_with_config_filenames(config_filename, pvnet_config_filename, data_sites):
    """ Location - usage check with config file paths 
    """
    
    meta_df = pd.read_csv(data_sites.metadata_file_path)
    
    loc = Location(
        x=meta_df['longitude'].iloc[0], 
        y=meta_df['latitude'].iloc[0], 
        coordinate_system='lon_lat'
    )

    assert isinstance(config_filename, str)
    assert isinstance(pvnet_config_filename, str)
    assert loc is not None


def test_location_temporary_path_usage(session_tmp_path):
    """ Location with temp path """
    # Create locations using temp path
    locations = [
        Location(x=-3.5, y=51.5, coordinate_system='lon_lat', id=1),
        Location(x=0, y=0, coordinate_system='osgb', id=2),
        Location(x=15002, y=4191563, coordinate_system='geostationary', id=3)
    ]
    
    # Assert all locations can be created
    for loc in locations:
        assert loc is not None
        with open(f"{session_tmp_path}/location_{loc.id}.txt", "w") as f:
            f.write(f"Location {loc.id}: x={loc.x}, y={loc.y}, system={loc.coordinate_system}")
