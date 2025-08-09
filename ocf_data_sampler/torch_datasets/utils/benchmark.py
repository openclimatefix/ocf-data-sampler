"""Benchmarking utilities for comparing local vs Ice Chunk performance."""

import logging
import time
from typing import Any, Dict, List, Optional

from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.load.load_dataset import get_dataset_dict

logger = logging.getLogger(__name__)


class OCFDataSamplerBenchmark:
    """Benchmark OCF Data Sampler with local vs Ice Chunk performance."""
    
    def __init__(self, config_path: str):
        """Initialize benchmark with configuration."""
        self.config = load_yaml_configuration(config_path)
        self.config_path = config_path
        
    def benchmark_satellite_loading(self, 
                                   local_zarr_path: Optional[str] = None,
                                   num_samples: int = 5) -> Dict[str, Any]:
        """Benchmark satellite data loading: local vs Ice Chunk."""
        logger.info("Starting OCF Data Sampler satellite loading benchmark...")
        
        results = {
            'config_path': self.config_path,
            'num_samples': num_samples,
            'satellite_config': {}
        }
        
        if not self.config.input_data.satellite:
            raise ValueError("No satellite configuration found in config")
        
        sat_config = self.config.input_data.satellite
        results['satellite_config'] = {
            'channels': sat_config.channels
        }
        
        benchmark_results = []
        
        for sample_idx in range(num_samples):
            logger.info(f"Running benchmark sample {sample_idx + 1}/{num_samples}")
            
            sample_result = {
                'sample_idx': sample_idx + 1,
                'methods': {}
            }
            
            # Use local_zarr_path if provided, otherwise use path from config
            zarr_path_to_use = local_zarr_path or sat_config.zarr_path
            # Temporarily update config for the benchmark run if overriding path
            original_path = sat_config.zarr_path
            sat_config.zarr_path = zarr_path_to_use

            method_name = "icechunk" if ".icechunk" in str(zarr_path_to_use) else "zarr"
            sample_result['methods'][method_name] = self._benchmark_single_load(method=method_name)

            # Restore original config path
            sat_config.zarr_path = original_path

            benchmark_results.append(sample_result)
        
        results['samples'] = benchmark_results
        results['aggregate'] = self._calculate_aggregate_results(benchmark_results)
        
        return results
    
    def _benchmark_single_load(self, method: str) -> Dict[str, Any]:
        """Benchmark a single loading method."""
        start_time = time.time()
        
        try:
            datasets_dict = get_dataset_dict(self.config.input_data)
            data = datasets_dict.get('sat')
            
            load_time = time.time() - start_time
            data_size_mb = data.nbytes / (1024**2) if data is not None else 0
            throughput = data_size_mb / load_time if load_time > 0 else 0
            
            return {
                'success': True,
                'load_time': load_time,
                'data_size_mb': data_size_mb,
                'throughput_mb_s': throughput,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': method
            }
    
    def _calculate_aggregate_results(self, benchmark_results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all samples."""
        methods = {}
        
        for sample in benchmark_results:
            for method_name, method_result in sample['methods'].items():
                if method_name not in methods:
                    methods[method_name] = []
                methods[method_name].append(method_result)
        
        aggregate = {}
        for method_name, results in methods.items():
            successful_results = [r for r in results if r.get('success', False)]
            
            if successful_results:
                throughputs = [r['throughput_mb_s'] for r in successful_results]
                load_times = [r['load_time'] for r in successful_results]
                
                aggregate[method_name] = {
                    'success_rate': len(successful_results) / len(results),
                    'avg_throughput_mb_s': sum(throughputs) / len(throughputs),
                    'avg_load_time': sum(load_times) / len(load_times),
                    'min_throughput': min(throughputs),
                    'max_throughput': max(throughputs),
                    'total_samples': len(results)
                }
            else:
                aggregate[method_name] = {
                    'success_rate': 0,
                    'total_samples': len(results),
                    'errors': [r.get('error', 'Unknown') for r in results]
                }
        
        return aggregate
    

def run_ocf_benchmark(config_path: str, 
                     local_zarr_path: Optional[str] = None,
                     num_samples: int = 3) -> Dict[str, Any]:
    """Run OCF Data Sampler benchmark comparing local vs Ice Chunk."""
    benchmark = OCFDataSamplerBenchmark(config_path)
    return benchmark.benchmark_satellite_loading(local_zarr_path, num_samples)
