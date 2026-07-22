import yaml

from ocf_data_sampler.spatial import Location, lon_lat_to_geostationary_area_coords
from tests.conftest import UK_SAT_AREA


def test_make_valid_location_object():
    _ = Location(x=-1000.5, y=50000, coord_system="osgb")


def test_lon_lat_to_geostationary_area_coords_accepts_area_mapping():
    """Test lon_lat_to_geostationary_area_coords function accepts a mapping or string.
    
    i.e. we can have the "area" attribute in the satellite data as either a string or a dict
    """
    area_mapping = yaml.safe_load(UK_SAT_AREA)

    string_coords = lon_lat_to_geostationary_area_coords(0.0, 50.0, UK_SAT_AREA)
    mapping_coords = lon_lat_to_geostationary_area_coords(0.0, 50.0, area_mapping)

    assert mapping_coords == string_coords
