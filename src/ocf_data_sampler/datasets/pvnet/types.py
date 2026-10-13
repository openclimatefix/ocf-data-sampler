"""Types for PVNet dataset orchestration."""

from dataclasses import dataclass, field
from typing import Any, Generic, Required, TypedDict, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

TArray = TypeVar("TArray")
TSource = TypeVar("TSource")


class BaseSample(TypedDict, Generic[TArray], total=False):
    """Optional sources, features and metadata shared by samples and batches.

    Shapes describe one sample; collation adds a leading batch dimension.
    Payloads and features are float32. Channel order follows configuration.

    Optional metadata fields:
        - location_longitude, location_latitude: degrees east/north.
        - satellite_time_utc: (time,), float64 POSIX seconds.
        - satellite_x_geostationary, satellite_y_geostationary: (x,) and (y,)
          coordinate vectors, retaining source units and dtypes.
        - nwp[provider]["init_time_utc"]: scalar, float64 POSIX seconds.
        - nwp[provider]["step_hours"]: (step,), float32 forecast lead times in hours.
    """

    # Generation: (time,), normalised by capacity.
    generation_input: TArray
    generation_target: TArray

    # Capacity: (time,), MW.
    generation_input_capacity_mwp: TArray
    generation_target_capacity_mwp: TArray

    # Timestamps: (time,), float64 POSIX seconds.
    generation_input_time_utc: TArray
    generation_target_time_utc: TArray

    # (time, channel, x, y).
    satellite: TArray

    # Provider name -> (valid_time, channel, x, y).
    nwp: dict[str, TArray]

    # (features,), two features per cyclic embedding and one per linear embedding.
    t0_embedding: TArray

    # (time,), sine/cosine encodings.
    date_sin: TArray
    date_cos: TArray
    time_sin: TArray
    time_cos: TArray

    # (time,), solar angles scaled to [0, 1].
    solar_azimuth: TArray
    solar_elevation: TArray

    metadata: dict[str, Any]


class NumpySample(BaseSample[NDArray[Any]], total=False):
    """One forecast time and location, with NumPy arrays."""

    location_id: Required[int]
    # Forecast reference time, float64 POSIX seconds.
    t0: Required[np.float64]


class TensorBatch(BaseSample[torch.Tensor], total=False):
    """Collated samples with a leading batch dimension on each leaf.

    Concurrent batches follow the dataset's location order.
    """

    # (batch,), int64 location IDs.
    location_id: Required[torch.Tensor]
    # (batch,), float64 POSIX seconds.
    t0: Required[torch.Tensor]


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

    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    clip_min: NDArray[np.float32]
    clip_max: NDArray[np.float32]


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
