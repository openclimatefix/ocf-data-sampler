from ocf_data_samplefrom ocf_data_sampler.spatial import Location


def test_make_valid_location_object():
    Location(x=-1000.5, y=50000, coord_system="osgb")
