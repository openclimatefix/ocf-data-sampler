"""Types for PVNet dataset orchestration."""

from dataclasses import dataclass, field
from typing import Any, Generic, Required, TypedDict, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

TSampleArray = TypeVar("TSampleArray")
TSource = TypeVar("TSource")


class BaseSample(TypedDict, Generic[TSampleArray], total=False):
    """Optional sources, features and metadata shared by samples and batches.

    Shapes describe one sample; collation adds a leading batch dimension.
    Payloads and features are float32. Channel order follows configuration.

    Optional metadata fields:
        - location_longitude, location_latitude: degrees east/north.
        - satellite_time_utc: (time,), float64 POSIX seconds.
        - satellite_x_geostationary, satellite_y_geostationary: (x,) and (y,)
          coordinate vectors, retaining source units and dtypes.
        - nwp[nwp_name]["init_time_utc"]: scalar, float64 POSIX seconds.
        - nwp[nwp_name]["step_hours"]: (step,), float32 forecast lead times in hours.
    """

    # Generation: (time,), normalised by capacity.
    generation_input: TSampleArray
    generation_target: TSampleArray

    # Capacity: (time,), MW.
    generation_input_capacity_mwp: TSampleArray
    generation_target_capacity_mwp: TSampleArray

    # Timestamps: (time,), float64 POSIX seconds.
    generation_input_time_utc: TSampleArray
    generation_target_time_utc: TSampleArray

    # (time, channel, x, y).
    satellite: TSampleArray

    # Configured source name -> (valid_time, channel, x, y).
    nwp: dict[str, TSampleArray]

    # (features,), two features per cyclic embedding and one per linear embedding.
    t0_embedding: TSampleArray

    # (time,), sine/cosine encodings.
    date_sin: TSampleArray
    date_cos: TSampleArray
    time_sin: TSampleArray
    time_cos: TSampleArray

    # (time,), solar angles scaled to [0, 1].
    solar_azimuth: TSampleArray
    solar_elevation: TSampleArray

    metadata: dict[str, Any]


class NumpySample(BaseSample[NDArray[Any]], total=False):
    """One forecast time and location, with NumPy arrays."""

    location_id: Required[str]
    # Forecast reference time, float64 POSIX seconds.
    t0: Required[np.float64]


class TensorBatch(BaseSample[torch.Tensor], total=False):
    """Collated samples with a leading batch dimension on each leaf.

    Concurrent batches follow the dataset's location order.
    """

    # (batch,), string location IDs.
    location_id: Required[list[str]]
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

    id: str
    longitude: float
    latitude: float
    source_coordinates: SourceDict[Coordinate] = field(default_factory=dict)
