""" Script to compute normalisation constants from NWP data """

import xarray as xr
import numpy as np
import glob
import argparse

from ocf_data_sampler.load.nwp.providers.icon import open_icon_eu

# Add argument parser
parser = argparse.ArgumentParser(description='Compute normalization constants from NWP data')
parser.add_argument('--data-path', type=str, required=True,
                    help='Path pattern to zarr files (e.g., "/path/to/data/*.zarr.zip")')
parser.add_argument('--n-samples', type=int, default=2000,
                    help='Number of random samples to use (default: 2000)')

args = parser.parse_args()

zarr_files = glob.glob(args.data_path)
n_samples = args.n_samples

ds = open_icon_eu(zarr_files)

n_init_times = ds.sizes['init_time_utc']
n_lats = ds.sizes['latitude']
n_longs = ds.sizes['longitude']
n_steps = ds.sizes['step']

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

samples_stack = xr.concat(samples, dim='samples')


# variables = [
#     "alb_rad", "aswdifd_s", "aswdir_s", "cape_con", "clch", "clcl", "clcm", 
#     "clct", "h_snow", "omega", "pmsl", "relhum_2m", "runoff_g", "runoff_s",
#     "t", "t_2m", "t_g", "td_2m", "tot_prec", "u", "u_10m", "v", "v_10m",
#     "vmax_10m", "w_snow", "ww", "z0"
# ]
print(samples_stack)

available_channels = samples_stack.channel.values.tolist()
print("Available channels: ", available_channels)

ICON_EU_MEAN = {}
ICON_EU_STD = {}

for var in available_channels:
    if var not in available_channels:
        print(f"Warning: Variable '{var}' not found in the channel coordinate; skipping.")
        continue
    var_data = samples_stack.sel(channel=var)
    var_mean = float(var_data.mean().compute())
    var_std = float(var_data.std().compute())
    
    ICON_EU_MEAN[var] = var_mean
    ICON_EU_STD[var] = var_std
    
    print(f"Processed {var}: mean={var_mean:.4f}, std={var_std:.4f}")

print("\nMean values:")
print(ICON_EU_MEAN)
print("\nStandard deviations:")
print(ICON_EU_STD)

