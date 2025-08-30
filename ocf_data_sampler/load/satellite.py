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
from contextlib import contextmanager

from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)

logger = logging.getLogger(__name__)

OPTIMAL_BLOCK_SIZE_MB = 64
OPTIMAL_THREADS = 2

def open_sat_data(zarr_path: str | list[str], channels: list[str] | None = None) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types."""
    
    if isinstance(zarr_path, list | tuple):
        ds = open_zarrs(zarr_path, concat_dim="time")
    else:
        # Parse path components using Sol's regex approach
        path_info = _parse_zarr_path(zarr_path)
        
        match path_info:
            case {"protocol": protocol, "bucket": bucket, "prefix": prefix, "sha1": sha1} if prefix.endswith(".icechunk"):
                ds = _open_sat_data_icechunk(protocol, bucket, prefix, sha1)
            
            case {"protocol": _, "bucket": _, "prefix": _, "sha1": None}:
                #  this doesn't work for blosc2 
                #  use ds = xr.open_dataset(zarr_path, engine="zarr", chunks="auto") in the case of blosc2
                ds = open_zarr(zarr_path)
                
            case _:
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
        "x_geostationary": "f",  
        "y_geostationary": "f",  
    }
    
    for coord, expected_kind in coord_dtypes.items():
        actual_kind = data_array.coords[coord].dtype.kind
        if actual_kind != expected_kind:
            # Special handling for datetime64 - accept any datetime64 precision
            if expected_kind == "M" and actual_kind == "M":
                continue  # Both are datetime64, just different precisions
            raise TypeError(f"Coordinate {coord} should be {expected_kind}, not {actual_kind}")
    
    return data_array

@contextmanager
def _setup_optimal_environment():
    """Apply optimization settings for cloud data streaming with context management."""

    original_values = {}
    env_vars = {
        "GCSFS_CACHE_TIMEOUT": "3600",
        "GCSFS_BLOCK_SIZE": str(OPTIMAL_BLOCK_SIZE_MB * 1024 * 1024),
        "GCSFS_DEFAULT_CACHE_TYPE": "readahead",
        "GOOGLE_CLOUD_DISABLE_GRPC": "true"
    }
    
    # Store original environment values
    for key in env_vars:
        original_values[key] = os.environ.get(key)
        os.environ[key] = env_vars[key]
    
    # Store original dask config (THIS MUST BE DECLARED HERE)
    original_dask_config = dict(dask.config.config)
    
    dask.config.set({
        "distributed.worker.memory.target": 0.7,
        "array.chunk-size": "512MB",
        "distributed.comm.compression": None,
        "distributed.worker.threads": OPTIMAL_THREADS,
    })
    
    try:
        yield
    finally:
        # Restore original environment values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        # Restore original dask config
        try:
            dask.config.reset()
        except AttributeError:
            # Use the correctly named variable
            dask.config.set(original_dask_config)

def _parse_zarr_path(path: str) -> dict:
    """Parse a path into its components, supporting both local and cloud paths."""

    pattern = r"^(?:(?P<protocol>[\w]{2,6}):\/\/)?(?P<bucket>\/?[\w-]+)\/(?P<prefix>[\w*.\/-]+?)(?:@(?P<sha1>[\w]+))?$"
    match = re.match(pattern, path)
    if not match:
        raise ValueError(f"Invalid path format: {path}")
    
    components = match.groupdict()
    
    if components["sha1"] is not None and not components["prefix"].endswith(".icechunk"):
        raise ValueError("Commit syntax (@commit) not supported for non-icechunk stores")
    
    if components["protocol"] == "gs" and components["prefix"] is not None and "*" in components["prefix"]:
        raise ValueError("Wildcard (*) paths are not supported for GCP (gs://) URLs")
    
    return components

def _open_sat_data_icechunk(
    protocol: str | None, bucket: str, prefix: str, sha1: str | None
) -> xr.Dataset:
    """Open satellite data from an Ice Chunk repository with optimized settings."""
    
    # Get storage according to protocol
    if protocol is None:
        logger.info(f"Opening local Ice Chunk repository: {prefix}")
        storage = icechunk.local_filesystem_storage(prefix)
    elif protocol == "gs":
        logger.info(f"Opening Ice Chunk repository: {protocol}://{bucket}/{prefix}")
        with _setup_optimal_environment():  
            # Ensure proper trailing slash
            if not prefix.endswith('/'):
                prefix = prefix + '/'
                
            logger.info(f"Accessing Ice Chunk repository: {protocol}://{bucket}/{prefix}")
            storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    else:
        raise ValueError(f"Unsupported protocol: {protocol}")

    try:
        repo = icechunk.Repository.open(storage)
    except Exception as e:
        logger.error(f"Failed to open Ice Chunk repository at {protocol or 'local'}://{bucket or ''}/{prefix}")
        raise e

    try:
        if sha1:
            session = repo.readonly_session(snapshot_id=sha1)
        else:
            session = repo.readonly_session("main")
    except Exception as e:
        target = sha1 or "main"
        raise ValueError(f"Failed to open session for '{target}': {e}") from e
        
    ds = xr.open_zarr(session.store, consolidated=True, chunks="auto")

    # Convert Ice Chunk format to standard format
    if len(ds.data_vars) > 1:
        data_arrays = [ds[var] for var in sorted(ds.data_vars)]
        combined_da = xr.concat(data_arrays, dim="variable")
        combined_da = combined_da.assign_coords(variable=sorted(ds.data_vars))
        ds = xr.Dataset({"data": combined_da})

    return ds
