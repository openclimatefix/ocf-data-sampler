"""Script to compute normalisation constants from NWP data."""

import argparse
import glob
import logging

import numpy as np
import xarray as xr

from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add argument parser
parser = argparse.ArgumentParser(description="Compute normalization constants from NWP data")
parser.add_argument("--data-path", type=str, required=True,
                    help='Path pattern to zarr files (e.g., "/path/to/data/*.zarr.zip")')
parser.add_argument("--n-samples", type=int, default=2000,
                    help="Number of random samples to use (default: 2000)")

args = parser.parse_args()

zarr_files = glob.glob(args.data_path)
n_samples = args.n_samples

ds = open_icon_eu(zarr_files)

n_init_times = ds.sizes["init_time_utc"]
n_lats = ds.sizes["latitude"]
n_longs = ds.sizes["longitude"]
n_steps = ds.sizes["step"]

random_init_times = np.random.choice(n_init_times, size=n_samples, replace=True)
random_lats = np.random.choice(n_lats, size=n_samples, replace=True)
random_longs = np.random.choice(n_longs, size=n_samples, replace=True)
random_steps = np.random.choice(n_steps, size=n_samples, replace=True)

samples = []
for i in range(n_samples):
    sample = ds.isel(init_time_utc=random_init_times[i],
                    latitude=random_lats[i],
                    longitude=random_longs[i],
                    step=random_steps[i])
    samples.append(sample)

samples_stack = xr.concat(samples, dim="samples")


available_channels = samples_stack.channel.values.tolist()
logger.info("Available channels: %s", available_channels)

ICON_EU_MEAN = {}
ICON_EU_STD = {}

for var in available_channels:
    if var not in available_channels:
        logger.warning("Variable '%s' not found in the channel coordinate; skipping.", var)
        continue
    var_data = samples_stack.sel(channel=var)
    var_mean = float(var_data.mean().compute())
    var_std = float(var_data.std().compute())

    ICON_EU_MEAN[var] = var_mean
    ICON_EU_STD[var] = var_std

    logger.info("Processed %s: mean=%.4f, std=%.4f", var, var_mean, var_std)

logger.info("\nMean values:\n%s", ICON_EU_MEAN)
logger.info("\nStandard deviations:\n%s", ICON_EU_STD)

