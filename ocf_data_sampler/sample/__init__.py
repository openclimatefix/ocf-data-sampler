# __init__.py

from ocf_data_sampler.sample.base import SampleBase
from ocf_data_sampler.sample.uk_regional import PVNetSample


__all__ = [
    'SampleBase',
    'PVNetSample'
    ]
