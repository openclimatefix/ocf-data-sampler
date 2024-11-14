""" Loads all data sources """
import xarray as xr

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.load.gsp import open_gsp
from ocf_data_sampler.load.nwp import open_nwp
from ocf_data_sampler.load.satellite import open_sat_data
from ocf_data_sampler.load.site import open_site


def get_dataset_dict(config: Configuration) -> dict[str, dict[xr.DataArray]]:
    """Construct dictionary of all of the input data sources

    Args:
        config: Configuration file
    """

    in_config = config.input_data

    datasets_dict = {}

    # Load GSP data unless the path is None
    if in_config.gsp and in_config.gsp.zarr_path:
        da_gsp = open_gsp(zarr_path=in_config.gsp.zarr_path).compute()

        # Remove national GSP
        datasets_dict["gsp"] = da_gsp.sel(gsp_id=slice(1, None))

    # Load NWP data if in config
    if in_config.nwp:

        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in in_config.nwp.items():

            da_nwp = open_nwp(nwp_config.zarr_path, provider=nwp_config.provider)

            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if in_config.satellite:
        sat_config = config.input_data.satellite

        da_sat = open_sat_data(sat_config.zarr_path)

        da_sat = da_sat.sel(channel=list(sat_config.channels))

        datasets_dict["sat"] = da_sat

    if in_config.site:
        da_sites = open_site(in_config.site)
        datasets_dict["site"] = da_sites

    return datasets_dict
