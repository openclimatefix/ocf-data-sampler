"""Conversion from Xarray to NumpySample"""

from .datetime_features import encode_datetimes
from .generation import convert_generation_to_numpy_sample, GenerationSampleKey
from .nwp import convert_nwp_to_numpy_sample, NWPSampleKey
from .satellite import convert_satellite_to_numpy_sample, SatelliteSampleKey
from .sun_position import make_sun_position_numpy_sample