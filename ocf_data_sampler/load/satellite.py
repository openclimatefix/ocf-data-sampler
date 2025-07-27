"""Satellite loader."""

import logging
import os
import re
from typing import Optional

import dask
import icechunk
import numpy as np
import xarray as xr
from xarray_tensorstore import open_zarr

from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)
from .open_tensorstore_zarrs import open_zarrs

logger = logging.getLogger(__name__)

# Optimal values from research, hardcoded as per Sol's feedback
OPTIMAL_BLOCK_SIZE_MB = 64
OPTIMAL_THREADS = 2

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

def _parse_icechunk_path(path: str) -> dict:
    """Parse a path into its components, supporting both local and cloud paths."""
    # Sol's recommended regex pattern - handles optional protocol and wildcards
    pattern = r"^(?:(?P<protocol>[\w]{2,6}):\/\/)?(?P<bucket>[\w\/-]+)\/(?P<prefix>[\w*.\/-]+?)(?:@(?P<sha1>[\w]+))?$"
    match = re.match(pattern, path)
    if not match:
        # For simple local paths without protocol, handle them separately
        if not path.startswith(('gs://', 'http', 'https')):
            return {
                "protocol": None,
                "bucket": None,
                "prefix": path,
                "sha1": None
            }
        raise ValueError(f"Invalid path format: {path}")
    
    components = match.groupdict()
    return components

def _open_sat_data_icechunk(
    protocol: str | None, bucket: str, prefix: str, sha1: str | None
) -> xr.Dataset:
    """Open satellite data from an Ice Chunk repository with optimized settings."""
    logger.info(f"Opening Ice Chunk repository: {protocol}://{bucket}/{prefix}")
    _setup_optimal_environment()
    
    # Remove .icechunk/.ic suffix from prefix for the icechunk library
    if prefix.endswith(".icechunk"):
        prefix = prefix.removesuffix(".icechunk")
    elif prefix.endswith(".ic"):
        prefix = prefix.removesuffix(".ic")
    
    # Ensure proper trailing slash
    if not prefix.endswith('/'):
        prefix = prefix + '/'
    
    # Set up storage and open the repository
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    try:
        repo = icechunk.Repository.open(storage)
    except Exception as e:
        logger.error(f"Failed to open repository at {protocol}://{bucket}/{prefix}")
        raise e
    
    # Handle different cases for commit vs branch access
    if sha1:
        logger.info(f"Opening Ice Chunk commit: {sha1}")
        try:
            # Try version parameter first
            session = repo.readonly_session(version=sha1)
            logger.info("Successfully opened session using 'version' parameter")
        except (TypeError, AttributeError):
            # If commit doesn't exist, error out (per Sol's feedback - no fallback)
            raise ValueError(f"Commit {sha1} not found in repository")
    else:
        logger.info("Opening 'main' branch of Ice Chunk repository.")
        session = repo.readonly_session(branch="main")
    
    # Open the dataset from the Ice Chunk session store
    ds = xr.open_zarr(session.store, consolidated=True, chunks="auto")
    
    # Combine separate channel variables into a single dataset with channel dimension
    if len(ds.data_vars) > 1:
        # Data is stored with separate variables per channel - combine them
        data_arrays = [ds[var] for var in sorted(ds.data_vars)]
        combined_da = xr.concat(data_arrays, dim="channel")
        combined_da = combined_da.assign_coords(channel=sorted(ds.data_vars))
        
        # Create dataset with satellite_data variable (to match OCF format)
        ds = xr.Dataset({"satellite_data": combined_da})
        ds = ds.assign_coords(variable=("channel", sorted(ds.data_vars)))
    
    return ds

def get_single_sat_data(zarr_path: str) -> xr.Dataset:
    """Opens a single satellite zarr file with Ice Chunk support via match/case patterns.
    
    This function implements Sol's requested architecture consolidating all conditional 
    logic into match/case patterns for different path types.
    """
    # Parse path components using Sol's regex approach
    path_info = _parse_icechunk_path(zarr_path)
    
    # Sol's requested match/case pattern for path routing
    match path_info:
        case {"protocol": None, "bucket": _, "prefix": _, "sha1": sha1} if sha1 is not None:
            # Raise error if trying to use commit syntax for local file
            raise ValueError("Commit syntax (@commit) not supported for local files")
        
        case {"protocol": "gs", "bucket": _, "prefix": prefix, "sha1": _} if "*" in prefix:
            # Raise an error if a wildcard is used in a GCP path
            raise ValueError("Wildcard (*) paths are not supported for GCP (gs://) URLs")
        
        case {"protocol": protocol, "bucket": bucket, "prefix": prefix, "sha1": sha1} if ".icechunk" in prefix and protocol is not None:
            # Ice Chunk logic goes here
            ds = _open_sat_data_icechunk(protocol, bucket, prefix, sha1)
        
        case {"protocol": _, "bucket": _, "prefix": _, "sha1": None}:
            # Existing single zarr logic path here
            ds = open_zarr(zarr_path)
        
        case {"protocol": None, "bucket": _, "prefix": prefix, "sha1": None} if "*" in prefix:
            # Handle multi-zarr dataset for local files
            ds = xr.open_mfdataset(
                zarr_path,
                engine="zarr",
                concat_dim="time",
                combine="nested",
                chunks="auto",
                join="override",
            )
        
        case _:
            # Fallback to standard zarr opening
            ds = open_zarr(zarr_path)
    
    return ds

def open_sat_data(zarr_path: str | list[str]) -> xr.DataArray:
    """Lazily opens the zarr store and validates data types.
    
    This function maintains the original signature and behavior while supporting
    Ice Chunk repositories through the consolidated get_single_sat_data logic.
    """
    if isinstance(zarr_path, list):
        ds = open_zarrs(zarr_path, concat_dim="time")
    else:
        ds = get_single_sat_data(zarr_path)
    
    check_time_unique_increasing(ds.time)
    ds = ds.rename({"variable": "channel", "time": "time_utc"})
    ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
    ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")
    
    data_array = get_xr_data_array_from_xr_dataset(ds)
    
    # Validate data types directly in loading function
    if not np.issubdtype(data_array.dtype, np.number):
        raise TypeError(f"Satellite data should be numeric, not {data_array.dtype}")
    
    coord_dtypes = {
        "time_utc": np.datetime64,
        "channel": np.str_,
        "x_geostationary": np.floating,
        "y_geostationary": np.floating,
    }
    
    for coord, expected_dtype in coord_dtypes.items():
        if not np.issubdtype(data_array.coords[coord].dtype, expected_dtype):
            raise TypeError(f"Coordinate {coord} should be {expected_dtype}, not {data_array.coords[coord].dtype}")
    
    return data_array