"""Conversion from Xarray to NumpyBatch"""

from .datetime_features import make_datetime_numpy_batch
from .gsp import convert_gsp_to_numpy_batch, GSPBatchKey
from .nwp import convert_nwp_to_numpy_batch, NWPBatchKey
from .satellite import convert_satellite_to_numpy_batch, SatelliteBatchKey
from .sun_position import make_sun_position_numpy_batch
from .site import convert_site_to_numpy_batch

