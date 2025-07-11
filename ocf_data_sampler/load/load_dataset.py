"""Loads all data sources."""

import xarray as xr

from ocf_data_sampler.config import InputData
from ocf_data_sampler.load import open_gsp, open_nwp, open_sat_data, open_site


def get_dataset_dict(
    input_config: InputData,
    gsp_ids: list[int] | None = None,
) -> dict[str, dict[xr.DataArray] | xr.DataArray]:
    """Construct dictionary of all of the input data sources.

    Args:
        input_config: InputData configuration object
        gsp_ids: List of GSP IDs to load. If None, all GSPs are loaded (not National).
    """
    datasets_dict = {}

    # Load GSP data unless the path is None
    if input_config.gsp and input_config.gsp.zarr_path:

        da_gsp = open_gsp(
            zarr_path=input_config.gsp.zarr_path,
            boundaries_version=input_config.gsp.boundaries_version,
            public=input_config.gsp.public,
        ).compute()

        if gsp_ids is None:
            # Remove national (gsp_id=0)
            da_gsp = da_gsp.sel(gsp_id=slice(1, None))
        else:
            da_gsp = da_gsp.sel(gsp_id=gsp_ids)

        datasets_dict["gsp"] = da_gsp

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

    if input_config.site:
        da_sites = open_site(
            generation_file_path=input_config.site.file_path,
            metadata_file_path=input_config.site.metadata_file_path,
        )

        datasets_dict["site"] = da_sites

    return datasets_dict
