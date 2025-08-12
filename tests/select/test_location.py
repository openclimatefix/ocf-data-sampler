from ocf_data_sampler.select.location import Location


def test_make_valid_location_object():
    x, y = -1000.5, 50000
    _ = Location(x=x, y=y, coord_system="osgb")
