import numpy as np

from ocf_data_sampler.spatial import lon_lat_to_geostationary_area_coords
from tests.conftest import UK_SAT_AREA


def test_lon_lat_to_geostationary_area_coords_projection_only():
    """The projection is sufficient to transform coordinates without shape or extent."""
    coords = lon_lat_to_geostationary_area_coords(0.0, 50.0, UK_SAT_AREA)

    # The expected coords below are pinned from an earlier version of the
    # lon_lat_to_geostationary_area_coords() function which extracted the CRS through Pyresample.
    # Test against these numbers to make sure the transform to the EUMETSAT RSS projection is
    # maintained
    np.testing.assert_allclose(
        coords, (-636553.2521574589, 4540516.747321172), rtol=0, atol=0.001,
    )
