"""Location model with coordinate system validation."""

from pydantic import BaseModel, Field, model_validator

allowed_coordinate_systems = ["osgb", "lon_lat", "geostationary", "idx"]


class Location(BaseModel):
    """Represent a spatial location."""

    coordinate_system: str = Field(...,
        description="Coordinate system for the location must be lon_lat, osgb, or geostationary",
    )

    x: float = Field(..., description="x coordinate - i.e. east-west position")
    y: float = Field(..., description="y coordinate - i.e. north-south position")
    id: int | None = Field(None, description="ID of the location - e.g. GSP ID")

    @model_validator(mode="after")
    def validate_coordinate_system(self) -> "Location":
        """Validate 'coordinate_system'."""
        if self.coordinate_system not in allowed_coordinate_systems:
            raise ValueError(
                f"coordinate_system = {self.coordinate_system} "
                f"is not in {allowed_coordinate_systems}",
            )
        return self
