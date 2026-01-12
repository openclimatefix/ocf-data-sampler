"""Functions to convert batches to tensors and move them to a given device."""

import numpy as np
import torch

from ocf_data_sampler.numpy_sample.common_types import NumpyBatch, TensorBatch


def batch_to_tensor(batch: dict) -> dict:
    """Convert numpy arrays in batch to torch tensors.
    
    Handles special cases like timestamps and nested dictionaries.
    
    Args:
        batch: Dictionary potentially containing numpy arrays
        
    Returns:
        Dictionary with numpy arrays converted to tensors
    """
    result = {}
    
    for key, value in batch.items():
        if key == "t0":
            # Convert pandas Timestamps to seconds since epoch
            if isinstance(value, np.ndarray):
                # Array of timestamps
                if value.dtype == object and len(value) > 0:
                    # Pandas Timestamp objects
                    import pandas as pd
                    if isinstance(value[0], pd.Timestamp):
                        # Convert to seconds since epoch
                        seconds = np.array([ts.timestamp() for ts in value])
                        result[key] = torch.from_numpy(seconds).float()
                    else:
                        result[key] = torch.from_numpy(value)
                else:
                    result[key] = torch.from_numpy(value)
            elif hasattr(value, 'timestamp'):
                # Single pandas Timestamp
                result[key] = torch.tensor(value.timestamp()).float()
            else:
                result[key] = value
                
        elif isinstance(value, np.ndarray):
            # Avoid converting string or object arrays to tensors
            if value.dtype.kind in ("U", "S") or value.dtype == object:
                result[key] = value
            else:
                result[key] = torch.from_numpy(value)
            
        elif isinstance(value, dict):
            # Recursively handle nested dictionaries (like NWP)
            result[key] = batch_to_tensor(value)
            
        else:
            # Keep as is (scalars, lists, etc.)
            result[key] = value
    
    return result



def copy_batch_to_device(batch: TensorBatch, device: torch.device) -> TensorBatch:
    """Recursively copies tensors in nested dict to specified device.

    Args:
        batch: Nested dict with tensors to move
        device: Device to move tensors to

    Returns:
        A dict with tensors moved to the new device
    """
    batch_copy = {}

    for k, v in batch.items():
        if isinstance(v, dict):
            batch_copy[k] = copy_batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            batch_copy[k] = v.to(device)
        else:
            batch_copy[k] = v
    return batch_copy
