"""Helper function to find index positions."""

from typing import Any, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


def get_string_index(values: NDArray[np.str_], query: str) -> int:
    """Find a string's position in a one-dimensional array of unique strings.

    The array may be unsorted. Raise ValueError unless the string matches exactly once.

    Args:
        values: One-dimensional array of unique strings, in any order.
        query: String whose integer index to find in `values`.
    """
    indices = np.flatnonzero(values == query)
    if indices.size != 1:
        raise ValueError(f"Expected one match for {query!r}, found {indices.size}")
    return int(indices[0])


@overload
def get_indices_in_sorted_unique(values: NDArray[T], query: T) -> np.intp: ...


@overload
def get_indices_in_sorted_unique(values: NDArray[T], query: NDArray[T]) -> NDArray[np.intp]: ...


def get_indices_in_sorted_unique(
    values: NDArray[T],
    query: T | NDArray[T],
) -> np.intp | NDArray[np.intp]:
    """Get the integer index(es) of a value(s) within a sorted array.

    Args:
        values: Sorted 1D array with unique elements.
        query: Scalar or array of elements to find in `values`.
    """
    indices = np.searchsorted(values, query, side="left")

    if (indices >= values.size).any() or (values[indices] != query).any():
        raise ValueError(f"Not all values in {query} exist in array {values}")

    return indices


def assert_values_unique_increasing(values: NDArray[Any], label: str = "values") -> None:
    """Assert the values are unique and monotonically increasing."""
    if not (values[:-1] < values[1:]).all():
        raise ValueError(f"{label} must be strictly increasing")
