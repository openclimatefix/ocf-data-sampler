# utils.py

""" 
Util functions for sample operations
To refine and potentially integrate with SampleBase ???
"""

import numpy as np
import logging
from typing import List, Dict, Any, Type

from ocf_data_sampler.sample.base import SampleBase


logger = logging.getLogger(__name__)


def stack_samples(samples: List[SampleBase]) -> SampleBase:
    """ Stack multiple samples into single sample """
    
    if not samples:
        raise ValueError("Cannot stack empty list of samples")
        
    if not all(isinstance(s, type(samples[0])) for s in samples):
        raise TypeError("All samples must be of same type")
        
    stacked = type(samples[0])()
    
    # Obtain all keys from first sample
    keys = samples[0].keys()
    
    for key in keys:
        try:
            # Handle nested dict - NWP
            if isinstance(samples[0][key], dict):
                stacked[key] = {}
                sub_keys = samples[0][key].keys()
                for sub_key in sub_keys:
                    try:
                        arrays = [s[key][sub_key] for s in samples]
                        stacked[key][sub_key] = np.stack(arrays)
                    except Exception as sub_err:
                        logger.error(f"Error stacking nested key {key}/{sub_key}: {sub_err}")
                        raise
            else:
                # Handle flat array case
                arrays = [s[key] for s in samples]
                stacked[key] = np.stack(arrays)
        except Exception as key_err:
            logger.error(f"Error processing key {key}: {key_err}")
            raise
            
    return stacked


def merge_samples(samples: List[SampleBase]) -> SampleBase:
   """ Merge multiple samples into single sample """

   if not samples:
       raise ValueError("Cannot merge empty list of samples")
       
   if not all(isinstance(s, type(samples[0])) for s in samples):
       raise TypeError("All samples must be of same type")
       
   merged = type(samples[0])()
   
   # Merge all keys
   for sample in samples:
       for key, value in sample._data.items():
           try:
               if key in merged._data:
                   logger.warning(f"Key {key} already exists in merged sample")
               merged[key] = value
           except Exception as key_err:
               logger.error(f"Error merging key {key}: {key_err}")
               raise
           
   return merged


def convert_batch_to_sample(batch_dict: Dict[str, Any], sample_class: Type[SampleBase]) -> SampleBase:
   """ Convert batch dict to sample obj """

   sample = sample_class()
   
   for key, value in batch_dict.items():
       try:
           if isinstance(value, dict):
               sample[key] = {k: v for k, v in value.items()}
           else:
               sample[key] = value
       except Exception as key_err:
           logger.error(f"Error converting key {key}: {key_err}")
           raise
           
   return sample