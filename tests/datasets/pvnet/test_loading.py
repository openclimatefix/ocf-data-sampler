from copy import deepcopy

import numpy as np
import pyproj
import xarray as xr

from ocf_data_sampler.datasets.pvnet.loading import add_source_coordinates
from ocf_data_sampler.datasets.pvnet.types import Location
from tests.conftest import UK_SAT_AREA


def test_add_source_coordinates_different_geostationary_sources():
    """Satellite and cloudcasting centres must use their respective projections."""
    cloudcasting_area = deepcopy(UK_SAT_AREA)
    cloudcasting_area["msg_seviri_rss_3km"]["projection"]["lon_0"] = 0
    spatial_coords = np.array([0.0, 1.0])
    coords = {"x_geostationary": spatial_coords, "y_geostationary": spatial_coords}
    satellite = xr.DataArray(
        np.zeros((len(spatial_coords), len(spatial_coords)), dtype=np.float32), coords=coords,
        dims=("x_geostationary", "y_geostationary"), attrs={"area": UK_SAT_AREA},
    )
    cloudcasting = satellite.assign_attrs(area=cloudcasting_area)
    locations = [Location(id=1, longitude=0.0, latitude=50.0)]

    add_source_coordinates(
        locations, {"sat": satellite, "nwp": {"cloudcasting": cloudcasting}},
    )

    source_coordinates = locations[0].source_coordinates
    sat_coordinate = source_coordinates["sat"]
    cloud_coordinate = source_coordinates["nwp"]["cloudcasting"]
    for coordinate, area in ((sat_coordinate, UK_SAT_AREA), (cloud_coordinate, cloudcasting_area)):
        transformer = pyproj.Transformer.from_crs(
            4326, area["msg_seviri_rss_3km"]["projection"], always_xy=True,
        )
        np.testing.assert_allclose(
            (coordinate.x, coordinate.y), transformer.transform(0.0, 50.0), rtol=0, atol=0.001,
        )
        assert (coordinate.x_dim, coordinate.y_dim) == ("x_geostationary", "y_geostationary")
    assert sat_coordinate.x != cloud_coordinate.x
