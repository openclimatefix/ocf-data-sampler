#!/usr/bin/env python3
"""Benchmark script for Ice Chunk vs Local performance."""

import argparse
import json
import logging

from ocf_data_sampler.torch_datasets.utils.benchmark import run_ocf_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ice Chunk vs Local performance")
    parser.add_argument("--config", required=True, help="Path to OCF configuration YAML")
    parser.add_argument("--local-zarr", help="Optional local zarr path for comparison")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to average")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    logger.info(f"Running OCF Data Sampler benchmark with {args.samples} samples")
    logger.info(f"Config: {args.config}")
    if args.local_zarr:
        logger.info(f"Local zarr: {args.local_zarr}")
    
    results = run_ocf_benchmark(
        config_path=args.config,
        local_zarr_path=args.local_zarr,
        num_samples=args.samples
    )
    
    print("\n" + "="*60)
    print("OCF DATA SAMPLER BENCHMARK RESULTS")
    print("="*60)
    
    if 'assessment' in results and 'recommendations' in results['assessment']:
        print("\nRECOMMENDATIONS:")
        for rec in results['assessment']['recommendations']:
            print(f"   {rec}")
    
    if 'aggregate' in results:
        print("\nPERFORMANCE SUMMARY:")
        for method, stats in results['aggregate'].items():
            if stats.get('success_rate', 0) > 0:
                print(f"   {method}: {stats['avg_throughput_mb_s']:.2f} MB/s avg")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
