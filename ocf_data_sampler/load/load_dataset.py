""" Loads all data sources """
import xarray as xr
import pandas as pd

from ocf_data_sampler.config import InputData
from ocf_data_sampler.load import open_nwp, open_gsp, open_sat_data, open_site
from ocf_data_sampler.numpy_sample.sun_position import calculate_azimuth_and_elevation


def get_dataset_dict(input_config: InputData) -> dict[str, dict[xr.DataArray] | xr.DataArray]:
    """Construct dictionary of all of the input data sources

    Args:
        input_config: InputData configuration object
    """

    datasets_dict = {}

    # Load GSP data unless the path is None
    if input_config.gsp and input_config.gsp.zarr_path:
        da_gsp = open_gsp(zarr_path=input_config.gsp.zarr_path).compute()

        # Remove national GSP
        da_gsp = da_gsp.sel(gsp_id=slice(1, None))
        
        # Add solar position data if configured
        if input_config.solar_position and input_config.solar_position.apply_to_gsp:
            _add_solar_position_to_gsp(da_gsp)
            
        datasets_dict["gsp"] = da_gsp

    # Load NWP data if in config
    if input_config.nwp:
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in input_config.nwp.items():
            da_nwp = open_nwp(nwp_config.zarr_path, provider=nwp_config.provider)
            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))
            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if input_config.satellite:
        sat_config = input_config.satellite
        da_sat = open_sat_data(sat_config.zarr_path)
        da_sat = da_sat.sel(channel=list(sat_config.channels))
        datasets_dict["sat"] = da_sat

    if input_config.site:
        da_sites = open_site(
            generation_file_path=input_config.site.file_path,
            metadata_file_path=input_config.site.metadata_file_path,
        )
        
        # Add solar position data if configured
        if input_config.solar_position and input_config.solar_position.apply_to_site:
            _add_solar_position_to_site(da_sites)
            
        datasets_dict["site"] = da_sites

    return datasets_dict


def _add_solar_position_to_gsp(da_gsp: xr.DataArray) -> None:
    """Add solar position data to GSP DataArray in-place
    
    Args:
        da_gsp: The GSP DataArray to modify
    """
    # Extract coordinates and timestamps
    times = pd.DatetimeIndex(da_gsp.time_utc.values)
    
    # For each GSP, calculate and add solar position
    for gsp_id in da_gsp.gsp_id.values:
        # Get location for this GSP
        lon = float(da_gsp.sel(gsp_id=gsp_id).lon_osgb.values)
        lat = float(da_gsp.sel(gsp_id=gsp_id).lat_osgb.values)
        
        # Calculate solar position
        azimuth, elevation = calculate_azimuth_and_elevation(times, lon, lat)
        
        # Normalize
        azimuth = azimuth / 360
        elevation = elevation / 180 + 0.5
        
        # Add as new data variables to the DataArray
        da_gsp.loc[dict(gsp_id=gsp_id)]["solar_azimuth"] = (("time_utc"), azimuth)
        da_gsp.loc[dict(gsp_id=gsp_id)]["solar_elevation"] = (("time_utc"), elevation)


def _add_solar_position_to_site(da_site: xr.DataArray) -> None:
    """Add solar position data to Site DataArray in-place
    
    Args:
        da_site: The Site DataArray to modify
    """
    # Extract timestamps
    times = pd.DatetimeIndex(da_site.time_utc.values)
    
    # For each site, calculate and add solar position
    for site_id in da_site.site_id.values:
        # Get location for this site
        lon = float(da_site.sel(site_id=site_id).longitude.values)
        lat = float(da_site.sel(site_id=site_id).latitude.values)
        
        # Calculate solar position
        azimuth, elevation = calculate_azimuth_and_elevation(times, lon, lat)
        
        # Normalize
        azimuth = azimuth / 360
        elevation = elevation / 180 + 0.5
        
        # Add as new data variables to the DataArray
        da_site.loc[dict(site_id=site_id)]["solar_azimuth"] = (("time_utc"), azimuth)
        da_site.loc[dict(site_id=site_id)]["solar_elevation"] = (("time_utc"), elevation)