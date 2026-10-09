from copy import deepcopy

import numpy as np
import pyproj
import xarray as xr

from ocf_data_sampler.datasets.pvnet.loading import add_source_coordinates
from ocf_data_sampler.datasets.pvnet.types import Location
from ocf_data_sampler.spatial import WGS84
from tests.conftest import UK_SAT_AREA


def test_add_source_coordinates_different_geostationary_sources():
    """Satellite and cloudcasting centres must use their respective projections."""

    # Make satellite-data
    coord_values = np.array([0.0, 1.0])
    da_sat = xr.DataArray(
        np.zeros((len(coord_values), len(coord_values))),
        coords={"x_geostationary": coord_values, "y_geostationary": coord_values},
        dims=("x_geostationary", "y_geostationary"),
        attrs={"area": UK_SAT_AREA},
    )

    # Make cloudcasting data that has a different satellite projection position. In this case set
    # the longitude to 0-degrees (the RSS (as in UK_SAT_AREA) is at 9.5 degrees)
    zero_degree_area = deepcopy(UK_SAT_AREA)
    zero_degree_area["msg_seviri_rss_3km"]["projection"]["lon_0"] = 0
    da_cloud = da_sat.copy().assign_attrs(area=zero_degree_area)

    # Make a location
    lon, lat = 0.0, 50.0
    location = Location(id=1, longitude=lon, latitude=lat)
    add_source_coordinates(
        locations=[location],
        datasets_dict={"sat": da_sat, "nwp": {"cloudcasting": da_cloud}}
    )

    # Each of the geostationary projections should be calculated independently
    sat_coord = location.source_coordinates["sat"]
    cloud_coord = location.source_coordinates["nwp"]["cloudcasting"]
    for coordinate, area in ((sat_coord, UK_SAT_AREA), (cloud_coord, zero_degree_area)):
        transformer = pyproj.Transformer.from_crs(
            WGS84, area["msg_seviri_rss_3km"]["projection"],
            always_xy=True,
        )
        np.testing.assert_allclose(
            (coordinate.x, coordinate.y), transformer.transform(lon, lat), rtol=0, atol=0.001,
        )
        assert (coordinate.x_dim, coordinate.y_dim) == ("x_geostationary", "y_geostationary")
    assert sat_coord.x != cloud_coord.x
