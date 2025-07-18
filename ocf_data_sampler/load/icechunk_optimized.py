"""Optimized cloud zarr loader for OCF Data Sampler."""


import logging
import time
import os
from typing import Any, Dict, List, Optional


import xarray as xr
import dask


from ocf_data_sampler.load.utils import (
    check_time_unique_increasing,
    get_xr_data_array_from_xr_dataset,
    make_spatial_coords_increasing,
)


logger = logging.getLogger(__name__)




class OptimizedIceChunkLoader:
    """Cloud zarr loader with streaming optimizations for OCF Data Sampler."""
   
    def __init__(self,
                 cloud_zarr_path: str,
                 bucket_name: str = "gsoc-dakshbir",
                 time_steps: int = 6,
                 block_size_mb: int = 64,
                 threads: int = 2):
        """Initialize optimized cloud zarr loader."""
        self.cloud_zarr_path = cloud_zarr_path.rstrip('/')
        self.bucket_name = bucket_name
        self.time_steps = time_steps
        self.block_size_mb = block_size_mb
        self.threads = threads
        self.spatial_x = None
        self.spatial_y = None
        self.dataset_info = {}
       
        self._setup_optimal_environment()
        logger.info(f"Initialized for gs://{bucket_name}/{cloud_zarr_path}")
   
    def _setup_optimal_environment(self):
        """Apply optimization settings."""
        logger.info("Applying optimization settings...")
       
        os.environ['GCSFS_CACHE_TIMEOUT'] = '3600'
        os.environ['GCSFS_BLOCK_SIZE'] = str(self.block_size_mb * 1024 * 1024)
        os.environ['GCSFS_DEFAULT_CACHE_TYPE'] = 'readahead'
        os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'
       
        dask.config.set({
            'distributed.worker.memory.target': 0.7,
            'array.chunk-size': '512MB',
            'distributed.comm.compression': None,
            'distributed.worker.threads': self.threads
        })
       
        logger.info("Optimization environment configured successfully")
   
    def _open_dataset_robust(self, dataset_path: str) -> xr.Dataset:
        """Robust dataset opening with multiple fallback methods."""
        try:
            return xr.open_zarr(dataset_path)
        except Exception as e1:
            logger.debug(f"Method 1 failed: {e1}")
       
        try:
            import fsspec
            mapper = fsspec.get_mapper(dataset_path, anon=False)
            return xr.open_zarr(mapper)
        except Exception as e2:
            logger.debug(f"Method 2 failed: {e2}")
       
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            store = gcsfs.mapping.GCSMap(dataset_path, gcs=fs)
            return xr.open_zarr(store)
        except Exception as e3:
            logger.debug(f"Method 3 failed: {e3}")
       
        try:
            import zarr
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            store = fs.get_mapper(dataset_path)
            zg = zarr.open_group(store)
            return xr.open_dataset(zg)
        except Exception as e4:
            logger.debug(f"Method 4 failed: {e4}")
       
        raise Exception("All dataset opening methods failed")
   
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Analyze dataset to determine optimal spatial dimensions."""
        logger.info(f"Analyzing dataset: {dataset_path}")
       
        try:
            ds = self._open_dataset_robust(dataset_path)
            ds_unique = ds.drop_duplicates('time')
            ds_sorted = ds_unique.sortby('time')
           
            available_vars = list(ds_sorted.variable.values) if 'variable' in ds_sorted.dims else []
            target_vars = [
                'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
                'VIS006', 'VIS008', 'WV_062', 'WV_073'
            ]
           
            if all(var in available_vars for var in target_vars):
                selected_vars = target_vars
                ds_channels = ds_sorted.sel(variable=selected_vars)
            else:
                selected_vars = available_vars[:3] if len(available_vars) >= 3 else available_vars
                ds_channels = ds_sorted.sel(variable=selected_vars) if selected_vars else ds_sorted
                logger.warning(f"Using available variables: {selected_vars}")
           
            self.spatial_x = ds_channels.sizes.get('x_geostationary', None)
            self.spatial_y = ds_channels.sizes.get('y_geostationary', None)
           
            if self.spatial_x and self.spatial_y:
                chunk_size_mb = (self.time_steps * self.spatial_x * self.spatial_y * len(selected_vars) * 2) / (1024**2)
            else:
                chunk_size_mb = 0
           
            self.dataset_info = {
                'path': dataset_path,
                'total_shape': dict(ds_channels.sizes),
                'spatial_x': self.spatial_x,
                'spatial_y': self.spatial_y,
                'variables': selected_vars,
                'time_steps_available': ds_channels.sizes.get('time', 0),
                'estimated_chunk_size_mb': chunk_size_mb,
                'total_chunks_possible': ds_channels.sizes.get('time', 0) // self.time_steps if ds_channels.sizes.get('time', 0) > 0 else 0
            }
           
            logger.info(f"Analysis complete: {self.spatial_x} × {self.spatial_y}, {len(selected_vars)} channels")
            return self.dataset_info
           
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            self.spatial_x = None
            self.spatial_y = None
            self.dataset_info = {}
            return {}
   
    def load_satellite_data_optimized(self,
                                    channels: Optional[List[str]] = None,
                                    spatial_slice: Optional[Dict[str, slice]] = None) -> xr.Dataset:
        """Load satellite data with optimizations."""
        cloud_path = f"gs://{self.bucket_name}/{self.cloud_zarr_path}"
       
        logger.info(f"Loading with optimizations: {cloud_path}")
        start_time = time.time()
       
        try:
            if not self.dataset_info:
                self.analyze_dataset(cloud_path)
           
            ds = self._open_dataset_robust(cloud_path)
           
            ds_unique = ds.drop_duplicates('time')
            ds_sorted = ds_unique.sortby('time')
           
            if channels and 'variable' in ds_sorted.dims:
                available_channels = list(ds_sorted.variable.values)
                selected_channels = [ch for ch in channels if ch in available_channels]
                if selected_channels:
                    ds_channels = ds_sorted.sel(variable=selected_channels)
                    logger.info(f"Selected channels: {selected_channels}")
                else:
                    ds_channels = ds_sorted
                    logger.warning(f"No matching channels found, using all: {available_channels}")
            else:
                ds_channels = ds_sorted
           
            if spatial_slice:
                for dim, slice_obj in spatial_slice.items():
                    if dim in ds_channels.dims:
                        ds_channels = ds_channels.isel({dim: slice_obj})
           
            load_time = time.time() - start_time
            data_size_mb = ds_channels.nbytes / (1024**2) if hasattr(ds_channels, 'nbytes') else 0
            throughput = data_size_mb / load_time if load_time > 0 else 0
           
            logger.info(f"Optimized loading complete: {load_time:.2f}s, {throughput:.2f} MB/s")
           
            return ds_channels
           
        except Exception as e:
            logger.error(f"Optimized loading failed: {e}")
            raise




def open_sat_data_icechunk_optimized(cloud_zarr_path: str,
                                   bucket_name: str = "gsoc-dakshbir",
                                   channels: Optional[List[str]] = None,
                                   time_steps: int = 6,
                                   block_size_mb: int = 64) -> xr.DataArray:
    """Open satellite data with cloud optimizations."""
    loader = OptimizedIceChunkLoader(
        cloud_zarr_path=cloud_zarr_path,
        bucket_name=bucket_name,
        time_steps=time_steps,
        block_size_mb=block_size_mb
    )
   
    ds = loader.load_satellite_data_optimized(channels=channels)
   
    if 'variable' in ds.dims:
        ds = ds.rename({"variable": "channel"})
    if 'time' in ds.dims:
        ds = ds.rename({"time": "time_utc"})
   
    if 'time_utc' in ds.dims:
        check_time_unique_increasing(ds.time_utc)
   
    if 'x_geostationary' in ds.dims and 'y_geostationary' in ds.dims:
        ds = make_spatial_coords_increasing(ds, x_coord="x_geostationary", y_coord="y_geostationary")
        ds = ds.transpose("time_utc", "channel", "x_geostationary", "y_geostationary")
   
    return get_xr_data_array_from_xr_dataset(ds)