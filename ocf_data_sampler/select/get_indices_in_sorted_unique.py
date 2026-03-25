"""Helper function to find index positions."""

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=np.generic)


def get_indices_in_sorted_unique(
    values: NDArray[T],
    query: T | NDArray[T],
) -> int | NDArray[np.intp]:
    """Get the integer index(es) of a value(s) within a sorted array.

    Args:
        values: Sorted 1D array with unique elements.
        query: Scalar or array of elements to find in `values`.
    """
    indices = np.searchsorted(values, query, side="left")

    if (indices >= values.size).any() or (values[indices] != query).any():
        raise ValueError(f"Not all values in {query} exist in array {values}")

    return indices
