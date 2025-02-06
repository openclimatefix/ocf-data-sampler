import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


def minutes(minutes: int | list[float]) -> pd.Timedelta | pd.TimedeltaIndex:
    """Timedelta minutes

    Args:
        minutes: the number of minutes, single value or list
    """
    return pd.to_timedelta(minutes, unit="m")


class Location(BaseModel):
    """Represent a spatial location"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    coordinate_system: Optional[str] = Field(
        default="osgb",
        description="Coordinate system")  # ["osgb", "lon_lat", "geostationary", "idx"]
    x: float
    y: float 
    id: Optional[int] = Field(default=None)

    @field_validator("coordinate_system", mode="before")
    def validate_coordinate_system(cls, v):
        """Validate 'coordinate_system'"""
        allowed = ["osgb", "lon_lat", "geostationary", "idx"]
        if v not in allowed:
            raise ValueError(f"coordinate_system = {v} is not in {allowed}")
        return v

    @field_validator("x")
    def validate_x(cls, v, values):
        """Validate 'x'"""
        co = values.data.get('coordinate_system')
        if not co:
            raise ValueError("coordinate_system must be specified")
        bounds = {
            "osgb": (-103976.3, 652897.98),
            "lon_lat": (-180, 180),
            "geostationary": (-5568748.275756836, 5567248.074173927),
            "idx": (0, float('inf'))
        }
        min_x, max_x = bounds[co]
        if v < min_x or v > max_x:
            raise ValueError(f"x = {v} must be within {[min_x, max_x]} for {co}")
        return v

    @field_validator("y")
    def validate_y(cls, v, values):
        """Validate 'y'"""
        co = values.data.get('coordinate_system')
        if not co:
            raise ValueError("coordinate_system must be specified")
        bounds = {
            "osgb": (-16703.87, 1199851.44),
            "lon_lat": (-90, 90),
            "geostationary": (1393687.2151494026, 5570748.323202133),
            "idx": (0, float('inf'))
        }
        min_y, max_y = bounds[co]
        if v < min_y or v > max_y:
            raise ValueError(f"y = {v} must be within {[min_y, max_y]} for {co}")
        return v
