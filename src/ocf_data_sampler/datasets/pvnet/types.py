"""Types for PVNet dataset orchestration."""

from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypedDict, TypeVar

import numpy as np
import torch

NumpySample: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
NumpyBatch: TypeAlias = dict[str, np.ndarray | dict[str, np.ndarray]]
TensorBatch: TypeAlias = dict[str, torch.Tensor | dict[str, torch.Tensor]]


TSource = TypeVar("TSource")

class SourceDict(TypedDict, Generic[TSource], total=False):
    """A fixed layout of optional source keys with values of a common type.

    NWP values are nested by source name.
    """

    sat: TSource
    generation: TSource
    generation_input: TSource
    generation_target: TSource
    nwp: dict[str, TSource]


@dataclass(eq=False)
class NormalisationArrays:
    """Broadcastable normalisation parameters for one source."""

    mean: np.ndarray
    std: np.ndarray
    clip_min: np.ndarray
    clip_max: np.ndarray


@dataclass
class Coordinate:
    """An x/y coordinate pair and its source dimension names."""

    x_dim: str
    y_dim: str
    x: float
    y: float


@dataclass
class Location:
    """A sampling location with source-specific coordinates."""

    id: int
    longitude: float
    latitude: float
    source_coordinates: SourceDict[Coordinate] = field(default_factory=dict)
