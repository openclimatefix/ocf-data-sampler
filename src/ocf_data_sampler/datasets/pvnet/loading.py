"""Loads all data sources."""

import logging

import numpy as np
import xarray as xr

from ocf_data_sampler.common.time_utils import minutes
from ocf_data_sampler.config import PVNetDataConfig
from ocf_data_sampler.datasets.pvnet.types import SourceDict
from ocf_data_sampler.load import open_generation, open_nwp, open_satellite
from ocf_data_sampler.load.conventions import validate_step_grid, validate_time_grid

logger = logging.getLogger(__name__)


def _warn_if_not_float32(source: str, data: xr.DataArray) -> None:
    if data.dtype != np.float32:
        logger.warning(
            f"{source} has dtype {data.dtype}; all data sources will be converted to float32 "
            "during materialisation",
        )


def get_dataset_dict(
    config: PVNetDataConfig,
    location_ids: list[int],
) -> SourceDict[xr.DataArray]:
    """Construct dictionary of all of the per-sample input data sources.

    Locations metadata is deliberately excluded - it isn't a per-sample source, so the caller
    loads it separately.

    Args:
        config: PVNetDataConfig configuration object
        location_ids: Requested generation location IDs, in selection order.
    """
    datasets_dict = {}

    # Load generation data if in config
    if config.generation is not None:
        da_gen = open_generation(zarr_path=config.generation.zarr_path)
        _warn_if_not_float32("generation", da_gen)

        validate_time_grid(
            times=da_gen["time_utc"].values,
            resolution=minutes(config.generation.time_resolution_minutes),
            source="generation",
        )

        missing = np.setdiff1d(location_ids, da_gen["location_id"].values)
        if len(missing) > 0:
            raise ValueError(f"Generation data is missing for location IDs: {missing}")

        datasets_dict["generation"] = da_gen.sel(location_id=location_ids)

    # Load NWP data if in config
    if config.nwp:
        datasets_dict["nwp"] = {}
        for nwp_source, nwp_config in config.nwp.items():
            da_nwp = open_nwp(zarr_path=nwp_config.zarr_path, provider=nwp_config.provider)
            _warn_if_not_float32(f"nwp/{nwp_source}", da_nwp)

            # The NWP init times and steps must be multiples of the configured resolution so that
            # the valid times are aligned to the configured resolution
            validate_time_grid(
                times=da_nwp["init_time_utc"].values,
                resolution=minutes(nwp_config.time_resolution_minutes),
                source=f"nwp/{nwp_source}",
            )

            validate_step_grid(
                steps=da_nwp["step"].values,
                resolution=minutes(nwp_config.time_resolution_minutes),
                source=f"nwp/{nwp_source}",
            )

            da_nwp = da_nwp.sel(channel=list(nwp_config.channels))

            datasets_dict["nwp"][nwp_source] = da_nwp

    # Load satellite data if in config
    if config.satellite:

        da_sat = open_satellite(config.satellite.zarr_path)
        _warn_if_not_float32("satellite", da_sat)

        validate_time_grid(
            times=da_sat["time_utc"].values,
            resolution=minutes(config.satellite.time_resolution_minutes),
            source="satellite",
        )

        da_sat = da_sat.sel(channel=list(config.satellite.channels))

        datasets_dict["sat"] = da_sat

    return datasets_dict
