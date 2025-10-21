"""Loads all data sources."""

import logging

import xarray as xr

from ocf_data_sampler.config import InputData
from ocf_data_sampler.load import open_generation, open_nwp, open_sat_data

logger = logging.getLogger(__name__)

def get_dataset_dict(
    input_config: InputData,
) -> dict[str, dict[xr.DataArray] | xr.DataArray]:
    """Construct dictionary of all of the input data sources.

    Args:
        input_config: InputData configuration object
    """
    datasets_dict = {}

    # Load GSP data unless the path is None
    if input_config.generation and input_config.generation.zarr_path:

        da_generation = open_generation(
            zarr_path=input_config.generation.zarr_path,
            public=input_config.generation.public,
        )

        da_generation = da_generation.sel(location_id=slice(1, None))

        datasets_dict["generation"] = da_generation
        logger.warning("If location ID 0 present this has been filtered out.")

    # Load NWP data if in config
    if input_config.nwp:
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in input_config.nwp.items():
            da_nwp = open_nwp(
                zarr_path=nwp_config.zarr_path,
                provider=nwp_config.provider,
                public=nwp_config.public,
            )

            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if input_config.satellite:
        sat_config = input_config.satellite

        da_sat = open_sat_data(sat_config.zarr_path)

        da_sat = da_sat.sel(channel=list(sat_config.channels))

        datasets_dict["sat"] = da_sat

    return datasets_dict
