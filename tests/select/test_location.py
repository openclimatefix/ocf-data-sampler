from ocf_data_sampler.select.location import Location


def test_make_valid_location_object():
    Location(x=-1000.5, y=50000, coord_system="osgb")
