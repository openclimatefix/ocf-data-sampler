"""Configuration model for the PVNet dataset."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field, field_validator, model_validator

from ocf_data_sampler.common.types import ZarrSource
from ocf_data_sampler.load.nwp import PROVIDER_REGISTRY


def _validate_grid_resolution_minutes(resolution: int) -> int:
    """Validate a positive cadence against the datetime floor/ceil restrictions.

    Cadences must be a factor or multiple of one hour and a factor of one day. Other cadences are
    currently unsupported by the shared floor/ceil helpers.
    """
    if 60 % resolution != 0 and resolution % 60 != 0:
        raise ValueError("Resolution in minutes must be a factor or multiple of 60")
    if 1440 % resolution != 0:
        raise ValueError("Resolution in minutes must be a factor of 1440 (one day)")
    return resolution


def _validate_dropout_interval_end(
    dropout_minute_probabilities: dict[int, float] | None,
    interval_end_minutes: int,
) -> None:
    """Reject choices that would leave every observation in the window untouched."""
    if dropout_minute_probabilities is None:
        return
    for delay_minutes in dropout_minute_probabilities:
        if delay_minutes <= -interval_end_minutes:
            raise ValueError(
                f"Dropout minutes ({delay_minutes}) must mask at least one value in the "
                f"selected interval ending at {interval_end_minutes} minutes relative to t0",
            )


def _validate_probability_sum(probabilities: dict[int, float]) -> dict[int, float]:
    """Require a valid total probability for dropout cutoffs or NWP delays."""
    if sum(probabilities.values()) > 1:
        raise ValueError("Probabilities must sum to at most 1")
    return probabilities


_MinuteProbabilities = Annotated[
    dict[Annotated[int, Field(ge=0)], Annotated[float, Field(gt=0, le=1)]],
    Field(min_length=1),
    AfterValidator(_validate_probability_sum),
]

_Embedding = tuple[
    Annotated[str, Field(pattern=r"^[0-9]+[hy]$")],
    Literal["cyclic", "linear"],
]


def _validate_unique_channel_names(channels: Sequence["Channel"]) -> None:
    """Reject duplicate source channel names."""
    names = [channel.name for channel in channels]
    if len(names) != len(set(names)):
        raise ValueError("Channel names must be unique")


class Base(BaseModel, extra="forbid"):
    """Base configuration model that rejects unknown fields."""


class TimeWindowMixin(Base):
    """Time window bounds and temporal resolution in minutes."""

    time_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="Temporal resolution of the data",
    )

    interval_start_minutes: int = Field(
        ...,
        description="Interval starts at `t0 + interval_start_minutes` inclusive",
    )

    interval_end_minutes: int = Field(
        ...,
        description="Interval ends at `t0 + interval_end_minutes` inclusive",
    )

    @model_validator(mode="after")
    def validate_intervals(self) -> "TimeWindowMixin":
        """Require ordered interval bounds aligned to the temporal resolution."""
        start = self.interval_start_minutes
        end = self.interval_end_minutes
        resolution = self.time_resolution_minutes
        if start > end:
            raise ValueError(
                f"interval_start_minutes ({start}) must be <= interval_end_minutes ({end})",
            )
        if start % resolution != 0:
            raise ValueError(
                f"interval_start_minutes ({start}) must be divisible "
                f"by time_resolution_minutes ({resolution})",
            )
        if end % resolution != 0:
            raise ValueError(
                f"interval_end_minutes ({end}) must be divisible "
                f"by time_resolution_minutes ({resolution})",
            )
        return self


class FillValueMixin(Base):
    """Optional replacement value for missing data."""

    fill_value: float | None = Field(
        default=None,
        description="The value used to fill dropped observations and missing data. This is "
        "applied after normalisation, so it is in normalised units. None leaves NaNs unfilled.",
    )


class DropoutMixin(FillValueMixin):
    """Optional observation masking with a required fill value when enabled."""

    dropout_minute_probabilities: _MinuteProbabilities | None = Field(
        default=None,
        description="Minutes before t0 mapped to dropout probabilities. Values after the chosen "
        "cutoff are masked and filled using fill_value. Each cutoff must precede the "
        "interval end. Remaining probability means no dropout; None disables dropout.",
    )

    @model_validator(mode="after")
    def validate_fill_value(self) -> "DropoutMixin":
        """Require a replacement value when observations can be dropped out."""
        if self.dropout_minute_probabilities is not None and self.fill_value is None:
            raise ValueError(
                "fill_value must be set when dropout_minute_probabilities is configured",
            )
        return self


class SpatialWindowMixin(Base):
    """Spatial window dimensions in pixels."""

    image_size_pixels_height: int = Field(
        ...,
        gt=0,
        description="Height of the region of interest in pixels",
    )

    image_size_pixels_width: int = Field(
        ...,
        gt=0,
        description="Width of the region of interest in pixels",
    )


class Channel(Base):
    """A source channel and its normalisation parameters."""

    name: str = Field(..., description="Channel name in the source data")

    mean: float = Field(..., description="Mean used for normalisation")
    std: float = Field(..., gt=0, description="Standard deviation used for normalisation")
    clip_min: float | None = Field(
        None,
        description="Lower clipping bound before normalisation. None disables this bound.",
    )
    clip_max: float | None = Field(
        None,
        description="Upper clipping bound before normalisation. None disables this bound.",
    )

    @model_validator(mode="after")
    def validate_clip_range(self) -> "Channel":
        """Require the minimum clipping bound to be below the maximum when both are set."""
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min >= self.clip_max
        ):
            raise ValueError(
                f"clip_min ({self.clip_min}) must be less than clip_max ({self.clip_max})",
            )
        return self


class NWPChannel(Channel):
    """A source channel with normalisation parameters and optional differencing.

    Clipping and normalisation apply to the differences when accumulated is True.
    """

    accumulated: bool = Field(
        default=False,
        description="Whether to difference the channel before normalisation. The normalisation "
        "values (`mean`, `std`, `min_clip`, and `max_clip`) apply to the differenced values. "
        "Each difference is labelled with the earlier step's timestamp, and the output channel "
        "name is prefixed with diff_.",
    )



class Satellite(TimeWindowMixin, DropoutMixin, SpatialWindowMixin):
    """Satellite configuration model."""

    zarr_path: ZarrSource = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[Channel] = Field(
        ...,
        min_length=1,
        description="Channels to include, in output order",
    )

    @field_validator("time_resolution_minutes")
    def validate_resolution(cls, resolution: int) -> int:
        """Validate the supported temporal resolution."""
        return _validate_grid_resolution_minutes(resolution)

    @model_validator(mode="after")
    def validate_dropout_interval_end(self) -> "Satellite":
        """Require each dropout choice to mask observations in the satellite window."""
        _validate_dropout_interval_end(self.dropout_minute_probabilities, self.interval_end_minutes)
        return self

    @field_validator("channels")
    def validate_channel_names(cls, channels: list[Channel]) -> list[Channel]:
        """Reject duplicate source channel names."""
        _validate_unique_channel_names(channels)
        return channels


class NWP(TimeWindowMixin, FillValueMixin, SpatialWindowMixin):
    """NWP configuration model."""

    zarr_path: ZarrSource = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[NWPChannel] = Field(
        ...,
        min_length=1,
        description="Channels to include, in output order",
    )

    delay_minute_probabilities: _MinuteProbabilities | None = Field(
        default=None,
        description="Minutes before t0 mapped to probabilities. Selects a forecast initialised "
        "at or before t0 minus the chosen delay. Remaining probability means no added delay; "
        "None disables delays.",
    )

    provider: str = Field(..., description="The provider of the NWP data")

    max_staleness_minutes: int | None = Field(
        None,
        description="Sets a limit on how stale an NWP init-time is allowed to be whilst still "
        "being used to construct an example. If set to None, then no additional limit is applied "
        "except that the forecast must cover the requested time interval.",
    )

    @field_validator("time_resolution_minutes")
    def validate_resolution(cls, resolution: int) -> int:
        """Validate the supported temporal resolution."""
        return _validate_grid_resolution_minutes(resolution)

    @field_validator("provider")
    def validate_provider(cls, provider: str) -> str:
        """Require a registered provider name."""
        if provider not in PROVIDER_REGISTRY:
            supported = ", ".join(sorted(PROVIDER_REGISTRY))
            raise ValueError(f"Unknown NWP provider {provider!r}. Supported: {supported}")
        return provider

    @field_validator("channels")
    def validate_channel_names(cls, channels: list[NWPChannel]) -> list[NWPChannel]:
        """Reject duplicate source channel names."""
        _validate_unique_channel_names(channels)
        return channels


class GenerationWindow(Base):
    """Interval bounds for a generation window.

    Unlike `TimeWindowMixin`, no temporal resolution here - it belongs to the shared
    generation data source (`Generation.time_resolution_minutes`), not to an individual window.
    """

    interval_start_minutes: int = Field(
        ...,
        description="Interval starts at `t0 + interval_start_minutes` inclusive",
    )

    interval_end_minutes: int = Field(
        ...,
        description="Interval ends at `t0 + interval_end_minutes` inclusive",
    )

    @model_validator(mode="after")
    def validate_interval_order(self) -> "GenerationWindow":
        """Require the interval start to be at or before its end."""
        start = self.interval_start_minutes
        end = self.interval_end_minutes
        if start > end:
            raise ValueError(
                f"interval_start_minutes ({start}) must be <= interval_end_minutes ({end})",
            )
        return self


class GenerationInputWindow(GenerationWindow, DropoutMixin):
    """A generation window with dropout configuration.

    Dropout is configurable here since only the input window (not the prediction target)
    should ever be randomly masked out.
    """

    @model_validator(mode="after")
    def validate_dropout_interval_end(self) -> "GenerationInputWindow":
        """Require each dropout choice to mask observations in the generation input window."""
        _validate_dropout_interval_end(self.dropout_minute_probabilities, self.interval_end_minutes)
        return self


class GenerationTargetWindow(GenerationWindow):
    """Generation target window configuration model."""


class Generation(Base):
    """Generation configuration model."""

    zarr_path: str = Field(
        ...,
        description="Absolute or relative zarr filepath. Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    time_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="The temporal resolution of the generation data in minutes. Must be a "
        "factor or multiple of 60 and a factor of 1440 (one day).",
    )

    input: GenerationInputWindow | None = None
    target: GenerationTargetWindow | None = None

    @field_validator("time_resolution_minutes")
    def validate_resolution(cls, resolution: int) -> int:
        """Validate the supported temporal resolution."""
        return _validate_grid_resolution_minutes(resolution)

    @model_validator(mode="after")
    def validate_windows(self) -> "Generation":
        """Validate the input/target windows are set and divisible by the shared resolution."""
        if self.input is None and self.target is None:
            raise ValueError(
                "At least one of `generation.input` or `generation.target` must be configured",
            )

        for name, window in (("input", self.input), ("target", self.target)):
            if window is None:
                continue
            for bound_name, bound in (
                ("interval_start_minutes", window.interval_start_minutes),
                ("interval_end_minutes", window.interval_end_minutes),
            ):
                if bound % self.time_resolution_minutes != 0:
                    raise ValueError(
                        f"generation.{name}.{bound_name} ({bound}) must be divisible by "
                        f"generation.time_resolution_minutes ({self.time_resolution_minutes})",
                    )
        return self


class SamplingGrid(Base):
    """Configuration for the (location, time) grid that t0 times are sampled from."""

    locations_csv_path: str = Field(
        ...,
        description="Absolute or relative CSV filepath to the locations metadata (location IDs "
        "and their coordinates) - see `ocf_data_sampler.load.locations.open_locations`.",
    )

    exclude_location_ids: list[int] = Field(
        default=[],
        description="Location IDs from the locations metadata to drop from the sampling grid. "
        "Every ID listed must be present in the locations data.",
    )

    t0_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="Spacing between candidate t0 times. Must be a multiple of the satellite and "
        "generation resolutions, a factor or multiple of 60, and a factor of 1440 (one day).",
    )

    @field_validator("t0_resolution_minutes")
    def validate_resolution(cls, resolution: int) -> int:
        """Validate the supported temporal resolution."""
        return _validate_grid_resolution_minutes(resolution)

    @field_validator("exclude_location_ids")
    def validate_exclude_location_ids_unique(cls, v: list[int]) -> list[int]:
        """Reject repeated excluded location IDs."""
        duplicates = {i for i in v if v.count(i) > 1}
        if duplicates:
            raise ValueError(f"exclude_location_ids contains duplicates: {sorted(duplicates)}")
        return v


class SolarPosition(TimeWindowMixin):
    """Solar position configuration model."""


class DatetimeEncoding(TimeWindowMixin):
    """Datetime encoding configuration model."""


class T0Embedding(Base):
    """Configuration for the t0 time embedding."""

    embeddings: list[_Embedding] = Field(
        ...,
        description="Period and representation pairs, e.g. ('6h', 'cyclic') or ('1y', 'linear'). "
        "Periods support 1-24 hours or positive integer years. Each cyclic period adds two "
        "elements (sine and cosine) to the output vector; each linear period adds one "
        "element (fraction through the period).",
    )

    @field_validator("embeddings")
    def validate_embeddings(cls, embeddings: list[_Embedding]) -> list[_Embedding]:
        """Require positive embedding periods, with hours limited to 24."""
        for period, _ in embeddings:
            value = int(period[:-1])
            if value <= 0:
                raise ValueError("Embedding period must be > 0")
            if period[-1] == "h" and value > 24:
                raise ValueError("Embedding period in hours must be <= 24")

        return embeddings


class PVNetDataConfig(Base):
    """Sources, sampling grid, and features used to construct PVNet samples."""

    sampling_grid: SamplingGrid
    satellite: Satellite | None = None
    nwp: dict[str, NWP] | None = Field(default=None, min_length=1)
    generation: Generation | None = None
    solar_position: SolarPosition | None = None
    datetime_encoding: DatetimeEncoding | None = None
    t0_embedding: T0Embedding | None = None

    @model_validator(mode="after")
    def validate_t0_resolution_alignment(self) -> "PVNetDataConfig":
        """Require the t0 resolution to be multiple of generation and satellite time resolution."""
        resolution = self.sampling_grid.t0_resolution_minutes
        for name, source in (("satellite", self.satellite), ("generation", self.generation)):
            if source is not None and resolution % source.time_resolution_minutes != 0:
                raise ValueError(
                    f"sampling_grid.t0_resolution_minutes ({resolution}) must be divisible by "
                    f"{name}.time_resolution_minutes ({source.time_resolution_minutes})",
                )
        return self
