"""Indices identifying valid ordinary and concurrent PVNet samples."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _sanitise_index(idx: int, n_samples: int) -> int:
    """Validate an integer index and resolve negative indices."""
    if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
        raise TypeError(f"Sample indices must be integers, got {type(idx)!r}")

    index = int(idx)
    if index < -n_samples or index >= n_samples:
        raise IndexError(f"Index {idx} out of range for sample index of length {n_samples}")

    if index < 0:
        index += n_samples

    return index


@dataclass(frozen=True)
class SampleIndex:
    """Store valid samples as aligned one-dimensional time and location arrays.

    Each position identifies one ``(t0, location_id)`` pair. Fields cannot be rebound,
    but the arrays remain mutable; callers must preserve their alignment.
    """

    t0: NDArray[np.datetime64]
    location_id: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate the shape and alignment of the sample arrays."""
        if self.t0.ndim != 1 or self.location_id.ndim != 1:
            raise ValueError("Sample index arrays must be one-dimensional")
        if len(self.t0) != len(self.location_id):
            raise ValueError("Sample index arrays must have equal lengths")

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.t0)

    def __getitem__(self, idx: int) -> tuple[np.datetime64, np.int64]:
        """Return a sample's time and location, supporting negative integer indices."""
        index = _sanitise_index(idx, len(self))
        return self.t0[index], self.location_id[index]

    def contains(self, t0: np.datetime64, location_id: int) -> bool:
        """Return whether the time and location identify a valid sample."""
        return bool(np.any((self.t0 == t0) & (self.location_id == location_id)))


@dataclass(frozen=True)
class ConcurrentSampleIndex:
    """Store t0 times available at every requested location.

    The availability builder determines which times qualify. The field cannot be
    rebound, but the underlying array remains mutable.
    """

    t0: NDArray[np.datetime64]

    def __post_init__(self) -> None:
        """Validate that the timestamp array is one-dimensional."""
        if self.t0.ndim != 1:
            raise ValueError("Concurrent sample index array must be one-dimensional")

    def __len__(self) -> int:
        """Return the number of valid concurrent samples."""
        return len(self.t0)

    def __getitem__(self, idx: int) -> np.datetime64:
        """Return a sample's time, supporting negative integer indices."""
        return self.t0[_sanitise_index(idx, len(self))]

    def contains(self, t0: np.datetime64) -> bool:
        """Return whether the time identifies a valid concurrent sample."""
        return bool(np.any(self.t0 == t0))
