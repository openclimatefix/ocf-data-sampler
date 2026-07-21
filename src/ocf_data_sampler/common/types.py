"""The array contract shared by ``xarray.DataArray`` and ``LightDataArray``.

``DataArrayLike`` defines the duck-typed surface the sampling pipeline is
allowed to use on array objects. It is deliberately minimal: every member is
an obligation on ``LightDataArray`` and a promise that ``xarray.DataArray``
also satisfies. Do not add members without checking that both implementations
conform (see the protocol conformance test) — this is the array-compatibility
contract, not a convenience grab-bag.
"""

from collections.abc import Hashable, Sequence
from typing import Any, Protocol, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

# Types accepted as a single-dimension indexer in `.isel()` method below.
Indexer: TypeAlias = (
    int
    | slice
    | Sequence[int]
    | Sequence[bool]
    | NDArray[np.integer]
    | NDArray[np.bool_]
)


class DataArrayLike(Protocol):
    """Structural type for objects that behave like ``xarray.DataArray``.

    Satisfied by ``xarray.DataArray`` and ``LightDataArray``. Pipeline code
    that is generic over the array backend should be typed against this
    protocol — usually via ``TArray`` so the concrete type is preserved
    through the pipeline — rather than against either concrete class.
    """

    @property
    def dims(self) -> tuple[Hashable, ...]:
        """Dimension names in data order (``Hashable`` to match xarray's typing)."""

    @property
    def data(self) -> Any:  # noqa: ANN401 - the backend is intentionally open: numpy, TensorStore, dask, ...
        """The backing array: numpy, TensorStore, or any duck array."""

    @data.setter
    def data(self, value: NDArray[Any]) -> None:
        """Rebind the backing array; the shape must match the existing data."""

    @property
    def values(self) -> NDArray[Any]:
        """The data as a numpy array, loading/computing it if necessary."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the backing array."""

    def isel(
        self,
        indexers: dict[str, Indexer] | None = None,
        **indexers_kwargs: Any,  # noqa: ANN401 - must stay Any: xarray's isel has non-indexer
        # keyword params (drop, missing_dims) whose types conflict with any narrower
        # annotation here; see Indexer for the types actually passed.
    ) -> Self:
        """Select by integer position along named dimensions (see ``Indexer``)."""

    def __getitem__(self, key: str) -> Self:
        """Return the named coordinate as an array-like."""

    def load(self) -> Self:
        """Load the data into memory in place and return self."""

    def __len__(self) -> int:
        """Length of the leading dimension."""


# Used to type functions that return the same kind of DataArray-like object they receive.
TArray = TypeVar("TArray", bound=DataArrayLike)
