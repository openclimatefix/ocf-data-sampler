from typing import Any, Protocol, Self, TypeVar
from numpy.typing import NDArray


class DataArrayLike(Protocol):
    """A protocol for objects that behave like xarray.DataArray."""

    @property
    def dims(self) -> tuple[str, ...]:
        ...

    @property
    def data(self) -> Any:
        """Return the underlying NumPy or lazy backend array."""
        ...

    @data.setter
    def data(self, value: NDArray[Any]) -> None: 
        ...

    @property
    def values(self) -> NDArray[Any]:
        ...

    def isel(self, **indexers: Any) -> Self:
        ...

    def __getitem__(self, key: Any) -> Self:
        ...

    def load(self) -> Self:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def __len__(self) -> int:
        ...


TArray = TypeVar("TArray", bound=DataArrayLike)