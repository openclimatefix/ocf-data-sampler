from ocf_data_sampler.select.location import Location
import pytest


def test_make_valid_location_object_with_default_coordinate_system():
    x, y = -1000.5, 50000
    location = Location(x=x, y=y)
    assert location.x == x, "location.x value not set correctly"
    assert location.y == y, "location.x value not set correctly"
    assert (
        location.coordinate_system == "osgb"
    ), "location.coordinate_system value not set correctly"


def test_make_valid_location_object_with_osgb_coordinate_system():
    x, y, coordinate_system = 1.2, 22.9, "osgb"
    location = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert location.x == x, "location.x value not set correctly"
    assert location.y == y, "location.x value not set correctly"
    assert (
        location.coordinate_system == coordinate_system
    ), "location.coordinate_system value not set correctly"


def test_make_valid_location_object_with_lon_lat_coordinate_system():
    x, y, coordinate_system = 1.2, 1.2, "lon_lat"
    location = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert location.x == x, "location.x value not set correctly"
    assert location.y == y, "location.x value not set correctly"
    assert (
        location.coordinate_system == coordinate_system
    ), "location.coordinate_system value not set correctly"


def test_make_invalid_location_object_with_invalid_osgb_x():
    x, y, coordinate_system = 10000000, 1.2, "osgb"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"


def test_make_invalid_location_object_with_invalid_osgb_y():
    x, y, coordinate_system = 2.5, 10000000, "osgb"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"


def test_make_invalid_location_object_with_invalid_lon_lat_x():
    x, y, coordinate_system = 200, 1.2, "lon_lat"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"


def test_make_invalid_location_object_with_invalid_lon_lat_y():
    x, y, coordinate_system = 2.5, -200, "lon_lat"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"


def test_make_invalid_location_object_with_invalid_coordinate_system():
    x, y, coordinate_system = 2.5, 1000, "abcd"
    with pytest.raises(ValueError) as err:
        _ = Location(x=x, y=y, coordinate_system=coordinate_system)
    assert err.typename == "ValidationError"
