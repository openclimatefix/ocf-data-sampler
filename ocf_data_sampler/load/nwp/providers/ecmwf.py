"""ECMWF provider loaders."""

import xarray as xr

from ocf_data_sampler.load.nwp.providers.utils import open_zarr_paths
from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
    get_xr_data_array_from_xr_dataset
)

MEANS_PATH = "/home/alex/from-sol/europe.zarr"


def open_ifs(zarr_path: str | list[str], ensemble_member: int | None) -> xr.DataArray:
    """Opens the ECMWF IFS NWP data.

    Args:
        zarr_path: Path to the zarr to open

    Returns:
        Xarray DataArray of the NWP data
    """
    ds = open_zarr_paths(zarr_path)

    ds = ds.rename_vars({'cloud_cover_high': 'hcc', 'cloud_cover_low': 'lcc', 
                                         'cloud_cover_medium': 'mcc', 
                                         'cloud_cover_total': 'tcc',
                                         'downward_longwave_radiation_flux_gl': 'dlwrf', 
                                         'downward_shortwave_radiation_flux_gl': 'dswrf', 
                                         'downward_ultraviolet_radiation_flux_gl': 'duvrs', 
                                         'direct_shortwave_radiation_flux_gl': 'sr',
                                         'snow_depth_gl': 'sde', 'temperature_sl': 't2m', 
                                         'total_precipitation_rate_gl': 'prate', 'visibility_sl': 'vis', 
                                         'wind_u_component_100m': 'u100', 'wind_u_component_10m': 'u10', 
                                         'wind_u_component_200m': 'u200', 'wind_v_component_100m': 'v100', 
                                         'wind_v_component_10m': 'v10', 'wind_v_component_200m': 'v200'})
    ds = ds.to_dataarray(dim='variable')


    # WP6 hack
    if ensemble_member:
        ds = ds.rename_vars({'cloud_cover_low': 'lcc', 'direct_shortwave_radiation_flux_gl': 'sr', 
                            'downward_longwave_radiation_flux_gl': 'dlwrf',
                            'downward_shortwave_radiation_flux_gl': 'dswrf', 
                            'downward_ultraviolet_radiation_flux_gl': 'duvrs', 'temperature_sl': 't2m'})
        ds = ds.to_dataarray(dim='variable')
        ds = ds.sel(ensemble_member = ensemble_member)

        means_ds = open_zarr_paths(MEANS_PATH)
        means_ds = means_ds.rename_vars({'cloud_cover_high': 'hcc', 'cloud_cover_low': 'lcc', 
                                         'cloud_cover_medium': 'mcc', 
                                         'cloud_cover_total': 'tcc',
                                         'downward_longwave_radiation_flux_gl': 'dlwrf', 
                                         'downward_shortwave_radiation_flux_gl': 'dswrf', 
                                         'downward_ultraviolet_radiation_flux_gl': 'duvrs', 
                                         'direct_shortwave_radiation_flux_gl': 'sr',
                                         'snow_depth_gl': 'sde', 'temperature_sl': 't2m', 
                                         'total_precipitation_rate_gl': 'prate', 'visibility_sl': 'vis', 
                                         'wind_u_component_100m': 'u100', 'wind_u_component_10m': 'u10', 
                                         'wind_u_component_200m': 'u200', 'wind_v_component_100m': 'v100', 
                                         'wind_v_component_10m': 'v10', 'wind_v_component_200m': 'v200'})
        means_ds = means_ds.to_dataarray(dim='variable')
        means_ds = means_ds.sel(
            init_time = ds.init_time, 
            variable=['hcc', 'mcc', 'tcc', 'sde', 'u10', 'v10'],
            latitude=ds.latitude,
            longitude=ds.longitude,
            step=ds.step)
        
        ds = xr.concat([ds, means_ds], dim='variable')


    # Rename
    ds = ds.rename(
        {
            "init_time": "init_time_utc",
            "variable": "channel",
        }
    )

    # Check the timestamps are unique and increasing
    check_time_unique_increasing(ds.init_time_utc)

    # Make sure the spatial coords are in increasing order
    ds = make_spatial_coords_increasing(ds, x_coord="longitude", y_coord="latitude")

    ds = ds.transpose("init_time_utc", "step", "channel", "longitude", "latitude")
    
    # TODO: should we control the dtype of the DataArray?
    return get_xr_data_array_from_xr_dataset(ds)
