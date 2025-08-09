#!/usr/bin/env python3
"""Production benchmark comparison between Ice Chunk and Plain Zarr."""

import time
from pathlib import Path
from ocf_data_sampler.torch_datasets.utils.benchmark import run_ocf_benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WARMUP_SAMPLES = 2
BENCHMARK_SAMPLES = 5

def production_benchmark_comparison():
    """Compare Ice Chunk vs Plain Zarr with complete dataset."""
    
    print("Production Benchmark Comparison")
    print("=" * 50)
    
    configs = [
        {
            'name': 'Plain Zarr (Current Production)',
            'config': PROJECT_ROOT / 'test_plain_zarr_clean.yaml',
            'description': 'Optimized plain zarr streaming'
        },
        {
            'name': 'Ice Chunk (Complete Dataset)',
            'config': PROJECT_ROOT / 'production_icechunk_2024-02_config.yaml',
            'description': 'Complete dataset in Ice Chunk format'
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"Config: {config['config']}")
        
        try:
            start_time = time.time()

            print(f"Warming up with {WARMUP_SAMPLES} samples...")
            run_ocf_benchmark(
                config_path=str(config['config']),
                num_samples=WARMUP_SAMPLES
            )

            print(f"Benchmarking with {BENCHMARK_SAMPLES} samples...")
            result = run_ocf_benchmark(
                config_path=str(config['config']),
                num_samples=BENCHMARK_SAMPLES
            )
            
            test_time = time.time() - start_time
            
            if result and 'aggregate' in result:
                for method, stats in result['aggregate'].items():
                    if stats.get('success_rate', 0) > 0:
                        results[config['name']] = {
                            'throughput_mb_s': stats['avg_throughput_mb_s'],
                            'success_rate': stats['success_rate'],
                            'test_time': test_time,
                            'method': method
                        }
                        
                        print(f"SUCCESS: {stats['avg_throughput_mb_s']:.2f} MB/s")
                        break
            else:
                print(f"FAILED: No valid results")
                results[config['name']] = {'success': False, 'test_time': test_time}
                
        except Exception as e:
            print(f"ERROR: {e}")
            results[config['name']] = {'success': False, 'error': str(e)}
    
    # Analysis
    print(f"\nProduction Comparison Results")
    print("=" * 40)
    
    if len([r for r in results.values() if r.get('success', True)]) >= 2:
        plain_zarr = results.get('Plain Zarr (Current Production)', {})
        ice_chunk = results.get('Ice Chunk (Complete Dataset)', {})
        
        if plain_zarr.get('throughput_mb_s') and ice_chunk.get('throughput_mb_s'):
            plain_perf = plain_zarr['throughput_mb_s']
            ice_perf = ice_chunk['throughput_mb_s']
            
            print(f"Performance Comparison:")
            print(f"  Plain Zarr: {plain_perf:.2f} MB/s")
            print(f"  Ice Chunk: {ice_perf:.2f} MB/s")
            
            if ice_perf > plain_perf:
                ratio = ice_perf / plain_perf
                print(f"  Ice Chunk is {ratio:.2f}x FASTER")
                recommendation = "Ice Chunk recommended for performance"
            elif plain_perf > ice_perf:
                ratio = plain_perf / ice_perf
                print(f"  Plain Zarr is {ratio:.2f}x FASTER")
                recommendation = "Plain Zarr recommended for performance"
            else:
                print(f"  Similar performance")
                recommendation = "Choose based on features needed"
            
            print(f"\nRecommendation: {recommendation}")
    
    return results

if __name__ == "__main__":
    results = production_benchmark_comparison()
    print(f"\nProduction comparison complete!")
