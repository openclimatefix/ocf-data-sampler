"""location"""

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


allowed_coordinate_systems =["osgb", "lon_lat", "geostationary", "idx"]

class Location(BaseModel):
    """Represent a spatial location."""

    coordinate_system: Optional[str] = "osgb"  # ["osgb", "lon_lat", "geostationary", "idx"]
    x: float
    y: float
    id: Optional[int] = Field(None)

    @model_validator(mode='after')
    def validate_coordinate_system(self):
        """Validate 'coordinate_system'"""
        if self.coordinate_system not in allowed_coordinate_systems:
            raise ValueError(f"coordinate_system = {self.coordinate_system} is not in {allowed_coordinate_systems}")
        return self

    @model_validator(mode='after')
    def validate_x(self):
        """Validate 'x'"""
        min_x: float
        max_x: float

        co = self.coordinate_system
        if co == "osgb":
            min_x, max_x = -103976.3, 652897.98
        if co == "lon_lat":
            min_x, max_x = -180, 180
        if co == "geostationary":
            min_x, max_x = -5568748.275756836, 5567248.074173927
        if co == "idx":
            min_x, max_x = 0, np.inf
        if self.x < min_x or self.x > max_x:
            raise ValueError(f"x = {self.x} must be within {[min_x, max_x]} for {co} coordinate system")
        return self

    @model_validator(mode='after')
    def validate_y(self):
        """Validate 'y'"""
        min_y: float
        max_y: float

        co = self.coordinate_system
        if co == "osgb":
            min_y, max_y = -16703.87, 1199851.44
        if co == "lon_lat":
            min_y, max_y = -90, 90
        if co == "geostationary":
            min_y, max_y = 1393687.2151494026, 5570748.323202133
        if co == "idx":
            min_y, max_y = 0, np.inf
        if self.y < min_y or self.y > max_y:
            raise ValueError(f"y = {self.y} must be within {[min_y, max_y]} for {co} coordinate system")
        return self
