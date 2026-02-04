"""Location object."""

allowed_coord_systems = {"osgb", "lon_lat", "geostationary"}


class Location:
    """A spatial location."""

    def __init__(self, x: float, y: float, coord_system: str, id: int | str | None = None) -> None:
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
            raise ValueError(f"Coordinate {coord_system} is not supported")

    def in_coord_system(self, coord_system: str) -> tuple[float, float]:
        """Get the location in a specified coordinate system.

        Args:
            coord_system: The desired output coordinate system
        """
        self._check_valid_coord_system(coord_system)

        if coord_system in self._projections:
            return self._projections[coord_system]

        raise ValueError(
            f"Requested the coordinate in {coord_system}. This has not yet been added. "
            "The current available coordinate systems are "
            f"{list(self._projections.keys())}",
        )

    def add_coord_system(self, x: float, y: float, coord_system: str) -> None:
        """Add the equivalent location in a different coordinate system.

        Args:
            x: The east-west / left-right coordinate
            y: The south-north / down-up coordinate
            coord_system: The coordinate system name
        """
        self._check_valid_coord_system(coord_system)
        if coord_system in self._projections:
            if (x, y) != self._projections[coord_system]:
                raise ValueError(
                    f"Tried to re-add coordinate projection {coord_system}, but the supplied "
                    f"coordinate values ({x}, {y}) do not match the already stored values "
                    f"{self._projections[coord_system]}",
                )
        else:
            self._projections[coord_system] = (x, y)

    @property
    def x(self) -> float:
        """Return the east-west coordinate (prefer `lon_lat` when available)."""
        if "lon_lat" in self._projections:
            return self._projections["lon_lat"][0]
        if "osgb" in self._projections:
            return self._projections["osgb"][0]
        return next(iter(self._projections.values()))[0]

    @property
    def y(self) -> float:
        """Return the north-south coordinate (prefer `lon_lat` when available)."""
        if "lon_lat" in self._projections:
            return self._projections["lon_lat"][1]
        if "osgb" in self._projections:
            return self._projections["osgb"][1]
        return next(iter(self._projections.values()))[1]

    # Provide aliases for clarity
    @property
    def longitude(self) -> float:
        """Alias for :pyattr:`x` (longitude)."""
        return self.x

    @property
    def latitude(self) -> float:
        """Alias for :pyattr:`y` (latitude)."""
        return self.y
