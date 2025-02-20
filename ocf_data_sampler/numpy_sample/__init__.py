"""Conversion from Xarray to NumpySample"""

from .datetime_features import make_datetime_numpy_dict
from .gsp import convert_gsp_to_numpy_sample, GSPSampleKey
from .nwp import convert_nwp_to_numpy_sample, NWPSampleKey
from .satellite import convert_satellite_to_numpy_sample, SatelliteSampleKey
from .sun_position import make_sun_position_numpy_sample
from .site import convert_site_to_numpy_sample

