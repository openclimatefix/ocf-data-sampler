# test_loading.py
import logging
import sys
from ocf_data_sampler.load import open_sat_data
from ocf_data_sampler.config import load_yaml_configuration

# Configure logging to see the info messages from the loaders
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# --- Define all channels ---
ALL_CHANNELS = [
    'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
    'VIS006', 'VIS008', 'WV_062', 'WV_073'
]

# --- CONFIGURATION ---
# The conversion script creates this file. Make sure this name matches.
CONFIG_FILE = "production_icechunk_2024-02_config.yaml"

def main():
    """Main test function."""
    try:
        config = load_yaml_configuration(CONFIG_FILE)
        icechunk_commit_path = config.input_data.satellite.zarr_path
    except FileNotFoundError:
        log.error(f"❌ FAILED: Config file '{CONFIG_FILE}' not found.")
        log.error("Please run 'scripts/full_dataset_icechunk_conversion.py' first to generate it.")
        sys.exit(1)
    except Exception as e:
        log.error(f"❌ FAILED: Could not load or parse config file '{CONFIG_FILE}'. Error: {e}")
        sys.exit(1)

    # The path from the config includes the commit SHA.
    # We can derive the path without the commit from it.
    icechunk_repo_path = icechunk_commit_path.split('@')[0]

    # --- Test Case 1: Standard Zarr Path ---
    # This should use the _open_sat_data_zarr helper function.
    plain_zarr_path = "gs://gsoc-dakshbir/2024-02_nonhrv.zarr"
    print("\n" + "="*50)
    print(f"TESTING: Standard Zarr loading for: {plain_zarr_path}")
    print("EXPECTED: 'Opening satellite data from standard Zarr...'")
    print("="*50)
    try:
        da_zarr = open_sat_data(
            zarr_path=plain_zarr_path,
            channels=ALL_CHANNELS
        )
        log.info(f"✅ SUCCESS: Loaded Zarr data with shape {da_zarr.shape}")
        log.info(f"   Channels: {da_zarr.channel.values.tolist()}")
    except Exception as e:
        log.error(f"❌ FAILED: Could not load Zarr data. Error: {e}")

    # --- Test Case 2: Ice Chunk Path (without commit) ---
    print("\n" + "="*50)
    print(f"TESTING: Ice Chunk loading for: {icechunk_repo_path}")
    print("EXPECTED: 'Opening 'main' branch of Ice Chunk repository...'")
    print("="*50)
    try:
        da_icechunk = open_sat_data(
            zarr_path=icechunk_repo_path,
            channels=ALL_CHANNELS
        )
        log.info(f"✅ SUCCESS: Loaded Ice Chunk data with shape {da_icechunk.shape}")
        log.info(f"   Channels: {da_icechunk.channel.values.tolist()}")
    except Exception as e:
        log.error(f"❌ FAILED: Could not load Ice Chunk data. Error: {e}")

    # --- Test Case 3: Ice Chunk Path with Commit SHA ---
    print("\n" + "="*50)
    print(f"TESTING: Ice Chunk loading with commit SHA: {icechunk_commit_path}")
    print("EXPECTED: 'Opening Ice Chunk commit: ...'")
    print("="*50)
    try:
        da_icechunk_commit = open_sat_data(
            zarr_path=icechunk_commit_path,
            channels=ALL_CHANNELS
        )
        log.info(f"✅ SUCCESS: Loaded Ice Chunk data from commit with shape {da_icechunk_commit.shape}")
        log.info(f"   Channels: {da_icechunk_commit.channel.values.tolist()}")
    except Exception as e:
        log.error(f"❌ FAILED: Could not load Ice Chunk data from commit. Error: {e}")

if __name__ == "__main__":
    main()
