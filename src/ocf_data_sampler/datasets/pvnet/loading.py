"""Loads all data sources."""

import xarray as xr

from ocf_data_sampler.config import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.load import open_generation, open_nwp, open_sat_data


def get_dataset_dict(config: PVNetDataConfig) -> SourceDict[xr.DataArray]:
    """Construct dictionary of all of the per-sample input data sources.

    Locations metadata is deliberately excluded - it isn't a per-sample source, so the caller
    loads it separately.

    Args:
        config: PVNetDataConfig configuration object
    """
    datasets_dict = {}

    # Load generation data if in config
    if config.generation is not None:
        datasets_dict["generation"] = open_generation(zarr_path=config.generation.zarr_path)

    # Load NWP data if in config
    if config.nwp:
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in config.nwp.items():
            da_nwp = open_nwp(zarr_path=nwp_config.zarr_path, provider=nwp_config.provider)

            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if config.satellite:

        da_sat = open_sat_data(config.satellite.zarr_path)

        da_sat = da_sat.sel(channel=list(config.satellite.channels))

        datasets_dict["sat"] = da_sat

    return datasets_dict
