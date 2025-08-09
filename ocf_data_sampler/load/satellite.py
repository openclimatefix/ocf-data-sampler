"""Satellite loader."""

import logging
import os
import re
from typing import List, Optional

import dask
import icechunk
import xarray as xr
from xarray_tensorstore import open_zarr
from ocf_data_sampler.load.open_tensorstore_zarrs import open_zarrs

from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)

logger = logging.getLogger(__name__)

# Optimal values from research, now hardcoded as per Sol's feedback.
OPTIMAL_BLOCK_SIZE_MB = 64
OPTIMAL_THREADS = 2

def open_sat_data(zarr_path: str | list[str], channels: Optional[List[str]] = None) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types."""
    
    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim="time")
    else:
        # Parse path components using Sol's regex approach
        path_info = _parse_zarr_path(zarr_path)
        
        # Sol's requested match/case pattern for path routing
        match path_info:
            case {"protocol": protocol, "bucket": bucket, "prefix": prefix, "sha1": sha1} if ".icechunk" in prefix and protocol is not None:
                # Ice Chunk logic goes here - Sol's requested signature
                ds = _open_sat_data_icechunk(protocol, bucket, prefix, sha1)
            
            case {"protocol": _, "bucket": _, "prefix": _, "sha1": None}:
                #  this doesn't work for blosc2 
                #  use ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto") in the case of blosc2
                ds = open_zarr(zarr_path)
                
            case _:
                # Raise error on unhandled path
                raise ValueError(f"Unhandled path format: {zarr_path}")

    check_time_unique_increasing(ds.time)
    
    # Select channels if provided (before renaming variables)
    if channels:
        ds = ds.sel(variable=channels)
    
    ds = ds.rename({"variable": "channel", "time": "time_utc"})
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")
    
    data_array = get_xr_data_array_from_xr_dataset(ds)
    
    # Validate data types directly in loading function
    if not data_array.dtype.kind in 'bifc':  # boolean, int, float, complex
        raise TypeError(f"Satellite data should be numeric, not {data_array.dtype}")
    
    # Updated coordinate validation - more flexible for datetime64 subtypes
    coord_dtypes = {
        "time_utc": "M",  # datetime64 (any precision)
        "channel": "U",   # Unicode string
        "x_geostationary": "f",  # floating
        "y_geostationary": "f",  # floating
    }
    
    for coord, expected_kind in coord_dtypes.items():
        actual_kind = data_array.coords[coord].dtype.kind
        if actual_kind != expected_kind:
            # Special handling for datetime64 - accept any datetime64 precision
            if expected_kind == "M" and actual_kind == "M":
                continue  # Both are datetime64, just different precisions
            raise TypeError(f"Coordinate {coord} should be {expected_kind}, not {actual_kind}")
    
    return data_array

def _setup_optimal_environment():
    """Apply optimization settings for cloud data streaming."""
    logger.debug("Applying optimization settings for cloud streaming...")

    os.environ["GCSFS_CACHE_TIMEOUT"] = "3600"
    os.environ["GCSFS_BLOCK_SIZE"] = str(OPTIMAL_BLOCK_SIZE_MB * 1024 * 1024)
    os.environ["GCSFS_DEFAULT_CACHE_TYPE"] = "readahead"
    os.environ["GOOGLE_CLOUD_DISABLE_GRPC"] = "true"

    dask.config.set(
        {
            "distributed.worker.memory.target": 0.7,
            "array.chunk-size": "512MB",
            "distributed.comm.compression": None,
            "distributed.worker.threads": OPTIMAL_THREADS,
        }
    )
    logger.debug("Optimization environment configured successfully")

def _parse_zarr_path(path: str) -> dict:
    """Parse a path into its components, supporting both local and cloud paths."""
    # Sol's recommended regex pattern - handles optional protocol and wildcards
    pattern = r"^(?:(?P<protocol>[\w]{2,6}):\/\/)?(?P<bucket>[\w\/-]+)\/(?P<prefix>[\w*.\/-]+?)(?:@(?P<sha1>[\w]+))?$"
    match = re.match(pattern, path)
    if not match:
        raise ValueError(f"Invalid path format: {path}")
    
    components = match.groupdict()
    
    # Validation checks moved from match block
    if components["protocol"] is None and components["sha1"] is not None:
        raise ValueError("Commit syntax (@commit) not supported for local files")
    
    if components["protocol"] == "gs" and components["prefix"] is not None and "*" in components["prefix"]:
        raise ValueError("Wildcard (*) paths are not supported for GCP (gs://) URLs")
    
    return components

def _open_sat_data_icechunk(
    protocol: str | None, bucket: str, prefix: str, sha1: str | None
) -> xr.Dataset:
    """Open satellite data from an Ice Chunk repository with optimized settings."""
    protocol_str = protocol or "unknown"
    logger.info(f"Opening Ice Chunk repository: {protocol_str}://{bucket}/{prefix}")
    _setup_optimal_environment()

    # Ensure proper trailing slash
    if not prefix.endswith('/'):
        prefix = prefix + '/'

    logger.info(f"Accessing Ice Chunk repository: gs://{bucket}/{prefix}")

    # Set up storage and open the repository
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    try:
        repo = icechunk.Repository.open(storage)
    except Exception as e:
        logger.error(f"Failed to open repository at gs://{bucket}/{prefix}")
        raise e

    # Handle different cases properly - NO FALLBACK as per Sol's feedback
    if sha1:
        logger.info(f"Opening Ice Chunk commit: {sha1}")
        try:
            # First verify the snapshot exists
            snapshot_info = repo.lookup_snapshot(sha1)
            logger.info(f"Found snapshot: {snapshot_info}")
            
            # Since the documentation shows readonly_session doesn't accept commit IDs,
            # and your debug shows this snapshot is the current main branch snapshot,
            # access via main branch but verify we get the correct snapshot
            session = repo.readonly_session("main")
            
            # Verify we got the expected snapshot
            if session.snapshot_id == sha1:
                logger.info(f"Successfully accessed commit {sha1} via main branch")
            else:
                raise ValueError(f"Expected snapshot {sha1}, but main branch has {session.snapshot_id}")
                
        except Exception as e:
            # NO FALLBACK - error out as Sol requested
            raise ValueError(f"Commit {sha1} not found or inaccessible: {e}")
    else:
        logger.info("Opening 'main' branch of Ice Chunk repository.")
        session = repo.readonly_session("main")

    # Open the dataset from the Ice Chunk session store
    ds = xr.open_zarr(session.store, consolidated=True, chunks="auto")

    # Convert Ice Chunk format to standard format
    if len(ds.data_vars) > 1:
        # Data is stored with separate variables per channel - combine them
        data_arrays = [ds[var] for var in sorted(ds.data_vars)]
        combined_da = xr.concat(data_arrays, dim="variable")
        combined_da = combined_da.assign_coords(variable=sorted(ds.data_vars))
        
        # Create dataset with data variable (to match OCF format)
        ds = xr.Dataset({"data": combined_da})

    return ds
