"""Location object."""


allowed_coord_systems = {"osgb", "lon_lat", "geostationary"}


class Location:
    """A spatial location."""

    def __init__(self, x: float, y: float, coord_system: int, id: int | str | None = None) -> None:
        """A spatial location.

        Args:
            x: The east-west / left-right location
            y: The south-north / down-up location
            coord_system: The coordinate system
            id: The location ID
        """
        self._check_valid_coord_system(coord_system)
        self._projections: dict[str, tuple[float, float]] = {coord_system: (x, y)}
        self.id = id

    @staticmethod
    def _check_valid_coord_system(coord_system: str) -> None:
        if coord_system not in allowed_coord_systems:
            raise ValueError(f"Coordinate  {coord_system} is not supported")

    def in_coord_system(self, coord_system: str) -> tuple[float, float]:
        """Get the location in a specified coordinate system.

        Args:
            coord_system: The desired output coordinate system
        """
        self._check_valid_coord_system(coord_system)

        if coord_system in self._projections:
            return self._projections[coord_system]
        else:
            raise ValueError(
                "Requested the coodinate in {coord_system}. This has not yet been added. "
                "The current available coordinate systems are "
                f"{list(self.self._projections.keys())}",
            )

    def add_coord_system(self, x: float, y: float, coord_system: int) -> None:
        """Add the equivalent location in a different coordinate system.

        Args:
            x: The east-west / left-right coordinate
            y: The south-north / down-up coordinate
            coord_system: The coordinate system name
        """
        self._check_valid_coord_system(coord_system)
        if coord_system in self._projections:
            if not (x, y)==self._projections[coord_system]:
                raise ValueError(
                    f"Tried to re-add coordinate projection {coord_system}, but the supplied"
                    f"coodrinate values ({x}, {y}) do not match the already stored values "
                    f"{self._projections[coord_system]}",
                )
        else:
            self._projections[coord_system] = (x, y)

