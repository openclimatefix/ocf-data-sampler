import pytest

from ocf_data_sampler.select.location import Location


def test_make_valid_location_object():
    x, y = -1000.5, 50000
    location = Location(x=x, y=y, coordinate_system="osgb")
    assert location.x == x, "location.x value not set correctly"
    assert location.y == y, "location.x value not set correctly"
    assert location.coordinate_system == "osgb", (
        "location.coordinate_system value not set correctly"
    )

def test_make_invalid_location_object_with_invalid_coordinate_system():
    x, y, coordinate_system = 2.5, 1000, "abcd"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"
