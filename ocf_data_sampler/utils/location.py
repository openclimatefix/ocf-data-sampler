"""location"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """Represent a spatial location."""

    coordinate_system: Optional[str] = "osgb"  # ["osgb", "lon_lat", "geostationary", "idx"]
    x: float
    y: float
    id: Optional[int] = Field(None)

    @validator("coordinate_system", pre=True, always=True)
    def validate_coordinate_system(cls, v):
        """Validate 'coordinate_system'"""
        allowed_coordinate_systen = ["osgb", "lon_lat", "geostationary", "idx"]
        if v not in allowed_coordinate_systen:
            raise ValueError(f"coordinate_system = {v} is not in {allowed_coordinate_systen}")
        return v

    @validator("x")
    def validate_x(cls, v, values):
        """Validate 'x'"""
        min_x: float
        max_x: float
        if "coordinate_system" not in values:
            raise ValueError("coordinate_system is incorrect")
        co = values["coordinate_system"]
        if co == "osgb":
            min_x, max_x = -103976.3, 652897.98
        if co == "lon_lat":
            min_x, max_x = -180, 180
        if co == "geostationary":
            min_x, max_x = -5568748.275756836, 5567248.074173927
        if co == "idx":
            min_x, max_x = 0, np.inf
        if v < min_x or v > max_x:
            raise ValueError(f"x = {v} must be within {[min_x, max_x]} for {co} coordinate system")
        return v

    @validator("y")
    def validate_y(cls, v, values):
        """Validate 'y'"""
        min_y: float
        max_y: float
        if "coordinate_system" not in values:
            raise ValueError("coordinate_system is incorrect")
        co = values["coordinate_system"]
        if co == "osgb":
            min_y, max_y = -16703.87, 1199851.44
        if co == "lon_lat":
            min_y, max_y = -90, 90
        if co == "geostationary":
            min_y, max_y = 1393687.2151494026, 5570748.323202133
        if co == "idx":
            min_y, max_y = 0, np.inf
        if v < min_y or v > max_y:
            raise ValueError(f"y = {v} must be within {[min_y, max_y]} for {co} coordinate system")
        return v
