from ocf_data_sampler.spatial import Location


def test_make_valid_location_object():
    _ = Location(x=-1000.5, y=50000, coord_system="osgb")
