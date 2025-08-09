#!/usr/bin/env python3
"""Ice Chunk dataset conversion tool."""

import icechunk
import xarray as xr
from datetime import datetime
import uuid
import time
import os
import dask

DATASET_TO_CONVERT = "2024-02_nonhrv.zarr"
BUCKET_NAME = "gsoc-dakshbir"

def setup_production_optimization():
    """Apply optimization settings for production conversion."""
    os.environ['GCSFS_CACHE_TIMEOUT'] = '7200'
    os.environ['GCSFS_BLOCK_SIZE'] = str(128 * 1024 * 1024)
    os.environ['GCSFS_DEFAULT_CACHE_TYPE'] = 'readahead'
    os.environ['GOOGLE_CLOUD_DISABLE_GRPC'] = 'true'
    
    dask.config.set({
        'distributed.worker.memory.target': 0.85,
        'distributed.worker.memory.spill': 0.95,
        'array.chunk-size': '2GB',
        'distributed.comm.compression': None,
        'distributed.worker.threads': 6,
        'array.slicing.split_large_chunks': True,
        'array.rechunk.method': 'p2p'
    })

def convert_dataset_to_icechunk(dataset_name, bucket_name):
    """Convert specified dataset to Ice Chunk format."""
    setup_production_optimization()
    
    dataset_path = f"gs://{bucket_name}/{dataset_name}"
    
    try:
        full_ds = xr.open_zarr(dataset_path)
        print(f"Dataset loaded: {dict(full_ds.sizes)}")
        print(f"Total size: {full_ds.nbytes / (1024**3):.2f} GB")
        
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return {'success': False, 'error': str(e)}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    base_name = dataset_name.replace('.zarr', '').replace('_nonhrv', '')
    icechunk_path = f"{base_name}_icechunk_full_{timestamp}_{unique_id}/"
    
    try:
        storage = icechunk.gcs_storage(bucket=bucket_name, prefix=icechunk_path, from_env=True)
        repo = icechunk.Repository.create(storage)
        session = repo.writable_session("main")
    except Exception as e:
        print(f"Failed to create Ice Chunk repository: {e}")
        return {'success': False, 'error': str(e)}
    
    total_time_steps = full_ds.sizes['time']
    batch_size = 300
    total_batches = (total_time_steps + batch_size - 1) // batch_size
    total_size_gb = full_ds.nbytes / (1024**3)
    
    conversion_start = time.time()
    
    try:
        for batch_idx, batch_start in enumerate(range(0, total_time_steps, batch_size)):
            batch_end = min(batch_start + batch_size, total_time_steps)
            
            print(f"Processing batch {batch_idx + 1}/{total_batches}")
            
            # Load the batch into memory to break the OCF-blosc2 codec dependency
            batch_ds = full_ds.isel(time=slice(batch_start, batch_end)).load()
            
            # Create a new dataset with each channel as a separate variable.
            # This is the critical change to match the high-performance plain Zarr structure
            # and eliminate the slow on-the-fly data transformation during loading.
            new_data_vars = {}
            for channel in batch_ds.variable.values:
                # Select data for one channel and drop the now-redundant 'variable' coordinate
                channel_data = batch_ds['data'].sel(variable=channel).drop_vars('variable')
                new_data_vars[channel] = channel_data
            
            # This dataset now has separate variables for IR_016, IR_039, etc.
            codec_free_batch = xr.Dataset(new_data_vars)
            
            # CRITICAL: Clear encoding on ALL variables and coordinates in the dataset.
            # This prevents any original codec (like ocf_blosc2 or blosc) from being
            # passed to the new Zarr store, which expects a default codec.
            for var in codec_free_batch.variables:
                codec_free_batch[var].encoding.clear()

            codec_free_batch.to_zarr(
                session.store,
                mode='a' if batch_start > 0 else 'w',
                region={'time': slice(batch_start, batch_end)} if batch_start > 0 else None,
                consolidated=False
            )
            
            del codec_free_batch
            del batch_ds
            
            time.sleep(0.5)
        
        import zarr
        zarr.consolidate_metadata(session.store)
        
        total_time = time.time() - conversion_start
        final_throughput = total_size_gb / (total_time / 60)
        
        commit_id = session.commit(f"Complete {dataset_name} - OCF-blosc2 to Ice Chunk conversion")
        
        print(f"Conversion complete!")
        print(f"Total data: {total_size_gb:.2f} GB")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Throughput: {final_throughput:.2f} GB/min")
        print(f"Repository: gs://{bucket_name}/{icechunk_path}")
        print(f"Commit ID: {commit_id}")
        
        return {
            'success': True,
            'icechunk_path': icechunk_path,
            'commit_id': commit_id,
            'original_dataset': dataset_name,
            'metrics': {
                'total_time_minutes': total_time / 60,
                'throughput_gb_min': final_throughput,
                'data_size_gb': total_size_gb,
                'batches_processed': total_batches
            }
        }
        
    except Exception as e:
        print(f"Conversion failed during processing: {e}")
        return {'success': False, 'error': str(e)}

def create_production_config(conversion_result):
    """Create production configuration for benchmarking."""
    if not conversion_result['success']:
        return None
    
    dataset_name = conversion_result['original_dataset'].replace('.zarr', '').replace('_nonhrv', '')
    config_filename = f"production_icechunk_{dataset_name}_config.yaml"
    
    all_channels = [
        'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 
        'VIS006', 'VIS008', 'WV_062', 'WV_073'
    ]
    
    normalisation_constants_str = ""
    for channel in all_channels:
        normalisation_constants_str += f"""
      {channel}:
        mean: 0.5
        std: 0.2"""

    repo_path_for_config = conversion_result['icechunk_path'].rstrip('/')

    config_content = f"""general:
  name: "Production Ice Chunk - {conversion_result['original_dataset']}"
  description: "Complete {conversion_result['original_dataset']} converted to Ice Chunk for production comparison"

input_data:
  satellite:
    zarr_path: "gs://{BUCKET_NAME}/{repo_path_for_config}.icechunk@{conversion_result['commit_id']}"
    channels: {all_channels}
    time_resolution_minutes: 15
    interval_start_minutes: -60
    interval_end_minutes: 0
    image_size_pixels_height: 128
    image_size_pixels_width: 128
    normalisation_constants:{normalisation_constants_str}"""
    
    with open(config_filename, 'w') as f:
        f.write(config_content)
    
    print(f"Configuration saved: {config_filename}")
    return config_filename

def main():
    print(f"Converting {DATASET_TO_CONVERT} to Ice Chunk format")
    
    result = convert_dataset_to_icechunk(DATASET_TO_CONVERT, BUCKET_NAME)
    
    if result['success']:
        config_file = create_production_config(result)
        print(f"Ready for benchmarking with config: {config_file}")
    else:
        print(f"Conversion failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
