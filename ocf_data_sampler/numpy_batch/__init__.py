"""Conversion from Xarray to NumpyBatch"""

from .gsp import convert_gsp_to_numpy_batch
from .nwp import convert_nwp_to_numpy_batch
from .satellite import convert_satellite_to_numpy_batch
from .sun_position import make_sun_position_numpy_batch

