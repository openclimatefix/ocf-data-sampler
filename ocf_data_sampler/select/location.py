"""Location object."""


allowed_coordinate_systems = {"osgb", "lon_lat", "geostationary"}


class Location:
    """A spatial location."""

    def __init__(self, x: float, y: float, coord_system: int, id: int | str | None = None):
        """A spatial location.
        
        Args:
            x: The east-west / left-right location
            y: The south-north / down-up location
            coord_system: The coordinate system
            id: The location ID
        """
        assert coord_system in allowed_coordinate_systems
        self._projections: dict[str, Location] = {coord_system: (x, y)}
        self.id = id

    def in_coord_system(self, coord_system: str) -> tuple[float, float]:
        """Get the location in a specified coordinate system.

        Args:
            coord_system: The desired output coordinate system
        """
        assert coord_system in allowed_coordinate_systems

        if coord_system in self._projections:
            return self._projections[coord_system]
        else:
            raise ValueError(
                "Requested the coodinate in {coord_system}. This has not yet been added. "
                "The current available coordinate systems are "
                f"{list(self.self._projections.keys())}"
            )

    def add_coord_system(self, x: float, y: float, coord_system: int) -> None:
        """Add the equivalent location in a different coordinate system

        Args:
            x: The east-west / left-right coordinate
            y: The south-north / down-up coordinate
            coord_system: The coordinate system name
        """
        assert coord_system in allowed_coordinate_systems
        if coord_system in self._projections:
            if not (x, y)==self._projections[coord_system]:
                raise ValueError(
                    f"Tried to re-add coordinate projection {coord_system}, but the supplied"
                    f"coodrinate values ({x}, {y}) do not match the already stored values "
                    f"{self._projections[coord_system]}"
                )
        else:
            self._projections[coord_system] = (x, y)

