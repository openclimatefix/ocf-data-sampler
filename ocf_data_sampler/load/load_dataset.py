"""Loads all data sources."""

import logging
import xarray as xr

from ocf_data_sampler.config import InputData
from ocf_data_sampler.load import open_gsp, open_nwp, open_sat_data, open_site

logger = logging.getLogger(__name__)

def get_dataset_dict(
    input_config: InputData,
    gsp_ids: list[int] | None = None,
) -> dict[str, dict[xr.DataArray] | xr.DataArray]:
    """Construct dictionary of all of the input data sources.

    Args:
        input_config: InputData configuration object
        gsp_ids: Optional list of GSP IDs to load. If None, loads all but national (gsp_id=0).
    """
    datasets_dict: dict[str, Any] = {}

    # Load GSP data if configured
    if input_config.gsp and input_config.gsp.zarr_path:
        da_gsp = open_gsp(
            zarr_path=input_config.gsp.zarr_path,
            boundaries_version=input_config.gsp.boundaries_version,
            public=input_config.gsp.public,
        ).compute()
        # remove ‘national’ GSP (gsp_id=0) if no explicit IDs passed
        if gsp_ids is None:
            da_gsp = da_gsp.sel(gsp_id=slice(1, None))
        else:
            da_gsp = da_gsp.sel(gsp_id=gsp_ids)
        datasets_dict["gsp"] = da_gsp

    # Load NWP data if configured
    if input_config.nwp:
        datasets_dict["nwp"] = {}
        for source, cfg in input_config.nwp.items():
            da_nwp = open_nwp(
                zarr_path=cfg.zarr_path,
                provider=cfg.provider,
                public=cfg.public,
            ).sel(channel=list(cfg.channels))
            datasets_dict["nwp"][source] = da_nwp

    # Load Satellite data if configured
    if input_config.satellite:
        sat_cfg = input_config.satellite
        use_icechunk = getattr(sat_cfg, "use_true_icechunk", False)

        if use_icechunk:
            logger.info(f"Using true Ice Chunk loader for: {sat_cfg.icechunk_path}")
            from ocf_data_sampler.load.icechunk_optimized_with_ics import (
                open_sat_data_icechunk_optimized as open_icechunk
            )
            da_sat = open_icechunk(
                cloud_zarr_path=sat_cfg.icechunk_path,
                bucket_name=sat_cfg.bucket_name,
                channels=list(sat_cfg.channels),
                time_steps=sat_cfg.optimal_time_steps,
                block_size_mb=sat_cfg.optimal_block_size_mb,
                use_true_icechunk=True,
                icechunk_branch=sat_cfg.icechunk_branch,
                icechunk_commit=sat_cfg.icechunk_commit,
            )
        elif sat_cfg.icechunk_path:
            logger.info(f"Using optimized cloud Zarr loader for: {sat_cfg.icechunk_path}")
            from ocf_data_sampler.load.icechunk_optimized import (
                open_sat_data_icechunk_optimized
            )
            da_sat = open_sat_data_icechunk_optimized(
                cloud_zarr_path=sat_cfg.icechunk_path,
                bucket_name=sat_cfg.bucket_name,
                channels=list(sat_cfg.channels),
                time_steps=sat_cfg.optimal_time_steps,
                block_size_mb=sat_cfg.optimal_block_size_mb,
            )
        elif sat_cfg.zarr_path:
            logger.info(f"Using traditional loader for: {sat_cfg.zarr_path}")
            da_sat = open_sat_data(sat_cfg.zarr_path)
        else:
            raise ValueError("No valid satellite data path found in configuration")

        da_sat = da_sat.sel(channel=list(sat_cfg.channels))
        datasets_dict["sat"] = da_sat

    # Load Site data if configured
    if input_config.site:
        da_sites = open_site(
            generation_file_path=input_config.site.file_path,
            metadata_file_path=input_config.site.metadata_file_path,
        )
        datasets_dict["site"] = da_sites

    return datasets_dict