"""Configuration model for the PVNet dataset."""

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator
from typing_extensions import override

from ocf_data_sampler.common.xr_tensorstore import ZarrSource
from ocf_data_sampler.load.nwp import PROVIDER_REGISTRY


class Base(BaseModel):
    """Pydantic Base model where no extras can be added."""

    model_config = ConfigDict(extra="forbid")


class TimeWindowMixin(Base):
    """Mixin class, to add interval start, end and resolution minutes."""

    time_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="The temporal resolution of the data in minutes",
    )

    interval_start_minutes: int = Field(
        ...,
        description="Data interval starts at `t0 + interval_start_minutes`",
    )

    interval_end_minutes: int = Field(
        ...,
        description="Data interval ends at `t0 + interval_end_minutes`",
    )

    @model_validator(mode="after")
    def validate_intervals(self) -> "TimeWindowMixin":
        """Validator for time interval fields."""
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
    """Mixin class, to add a value used for filling missing data."""

    dropout_fill_value: float = Field(
        default=0.0,
        description="The value used to fill in dropped out data or any missing values."
    )


class DropoutMixin(FillValueMixin):
    """Mixin class, to add dropout minutes."""

    dropout_timedeltas_minutes: list[int] = Field(
        default=[],
        description="List of possible minutes before t0 where data availability may start. Must be "
        "negative or zero.",
    )

    dropout_fraction: float | list[float] = Field(
        default=0.0,
        description="Either a float(Chance of dropout being applied to each sample) or a list of "
        "floats (probability that dropout of the corresponding timedelta is applied)",
    )

    @field_validator("dropout_timedeltas_minutes")
    def dropout_timedeltas_minutes_negative(cls, v: list[int]) -> list[int]:
        """Validate 'dropout_timedeltas_minutes'."""
        for m in v:
            if m > 0:
                raise ValueError("Dropout timedeltas must be negative")
        return v


    @field_validator("dropout_fraction")
    def dropout_fractions(cls, dropout_frac: float | list[float]) -> float | list[float]:
        """Validate 'dropout_frac'."""
        if isinstance(dropout_frac, float | int):
            if not (0<= dropout_frac <= 1):
                raise ValueError("Dropout fractions must be in range [0, 1]")

        elif isinstance(dropout_frac, list):
            if not dropout_frac:
                raise ValueError("List cannot be empty")

            if not all(0 <= i <= 1 for i in dropout_frac):
                raise ValueError("All dropout fractions must be in range [0, 1]")

            if not (0 <= sum(dropout_frac) <= 1):
                raise ValueError("The sum of dropout fractions must be in range [0, 1]")

        return dropout_frac


    @model_validator(mode="after")
    def dropout_instructions_consistent(self) -> "DropoutMixin":
        """Validator for dropout instructions."""
        if (
            isinstance(self.dropout_fraction, list)
            and len(self.dropout_fraction) != len(self.dropout_timedeltas_minutes)
        ):
            raise ValueError(
                "When `dropout_fraction` is a list, it must have the same length as "
                "`dropout_timedeltas_minutes`"
            )

        if self.dropout_fraction == 0:
            if self.dropout_timedeltas_minutes != []:
                raise ValueError("To use dropout timedeltas dropout fraction should be > 0")
        else:
            if self.dropout_timedeltas_minutes == []:
                raise ValueError("To dropout fraction > 0 requires a list of dropout timedeltas")
        return self


class SpatialWindowMixin(Base):
    """Mixin class, to add path and image size."""

    image_size_pixels_height: int = Field(
        ...,
        gt=0,
        description="The number of pixels of the height of the region of interest",
    )

    image_size_pixels_width: int = Field(
        ...,
        gt=0,
        description="The number of pixels of the width of the region of interest",
    )


class NormalisationValues(Base):
    """Normalisation parameters."""
    mean: float = Field(..., description="Mean value for normalization")
    std: float = Field(..., gt=0, description="Standard deviation (must be positive)")
    clip_min: float | None = Field(
        None,
        description="Minimum value to clip to before normalisation. If None, no clipping is "
        "applied",
    )
    clip_max: float | None = Field(
        None,
        description="Maximum value to clip to before normalisation. If None, no clipping is "
        "applied",
    )

    @model_validator(mode="after")
    def validate_clip_range(self) -> "NormalisationValues":
        """"Validate that if both clip_min and clip_max are provided, then clip_min < clip_max."""
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_min >= self.clip_max
        ):
            raise ValueError(
                f"clip_min ({self.clip_min}) must be less than clip_max ({self.clip_max})",
            )
        return self


class NormalisationConstantsMixin(Base):
    """Normalisation constants for multiple channels."""
    normalisation_constants: dict[str, NormalisationValues]


class Satellite(TimeWindowMixin, DropoutMixin, SpatialWindowMixin, NormalisationConstantsMixin):
    """Satellite configuration model."""

    zarr_path: ZarrSource = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[str] = Field(
        ...,
        description="the satellite channels that are used",
    )

    @model_validator(mode="after")
    def check_all_channel_have_normalisation_constants(self) -> "Satellite":
        """Check that all the channels have normalisation constants."""
        normalisation_channels = set(self.normalisation_constants.keys())
        missing_norm_values = set(self.channels) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalisation constants must be provided for all channels. Missing values for "
                f"channels: {missing_norm_values}",
            )
        return self


class NWP(TimeWindowMixin, DropoutMixin, SpatialWindowMixin, NormalisationConstantsMixin):
    """NWP configuration model."""

    zarr_path: ZarrSource = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[str] = Field(
        ...,
        description="the channels used in the nwp data",
    )

    provider: str = Field(..., description="The provider of the NWP data")

    accum_channels: list[str] = Field([], description="The NWP channels which need to be diffed")

    max_staleness_minutes: int | None = Field(
        None,
        description="Sets a limit on how stale an NWP init time is allowed to be whilst still being"
        " used to construct an example. If set to None, then the max staleness is set according to"
        " the maximum forecast horizon of the NWP and the requested forecast length.",
    )

    @model_validator(mode="after")
    def validate_accum_channels_subset(self) -> "NWP":
        """Validate accum_channels is subset of channels."""
        invalid_channels = set(self.accum_channels) - set(self.channels)
        if invalid_channels:
            raise ValueError(
                f"NWP provider '{self.provider}': all values in 'accum_channels' should "
                f"be present in 'channels'. Extra values found: {invalid_channels}",
            )
        return self

    @field_validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validator for 'provider'."""
        provider = v.lower()
        if provider not in PROVIDER_REGISTRY:
            supported = ", ".join(sorted(PROVIDER_REGISTRY))
            raise ValueError(f"Unknown NWP provider {v!r}. Supported: {supported}")
        return provider

    @model_validator(mode="after")
    def check_all_channel_have_normalisation_constants(self) -> "NWP":
        """Check that all the channels have normalisation constants."""
        normalisation_channels = set(self.normalisation_constants.keys())
        non_accum_channels = [c for c in self.channels if c not in self.accum_channels]
        accum_channel_names = [f"diff_{c}" for c in self.accum_channels]

        missing_norm_values = set(non_accum_channels) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalisation constants must be provided for all channels. Missing values for "
                f"channels: {missing_norm_values}",
            )

        missing_norm_values = set(accum_channel_names) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalisation constants must be provided for all channels. Accumulated "
                "channels which will be diffed require normalisation constant names which "
                "start with the prefix 'diff_'. The following channels were missing: "
                f"{missing_norm_values}.",
            )
        return self


class MultiNWP(RootModel):
    """Configuration for multiple NWPs."""

    root: dict[str, NWP]

    @override
    def __getattr__(self, item: str) -> NWP:
        return self.root[item]

    @override
    def __getitem__(self, item: str) -> NWP:
        return self.root[item]

    @override
    def __len__(self) -> int:
        return len(self.root)

    @override
    def __iter__(self) -> Iterator:
        return iter(self.root)

    def keys(self) -> Iterator[str]:
        """Returns dictionary-like keys."""
        return self.root.keys()

    def items(self) -> Iterator[tuple[str, NWP]]:
        """Returns dictionary-like items."""
        return self.root.items()


class GenerationWindow(Base):
    """Mixin class, to add interval start and end minutes for a generation window.

    Unlike `TimeWindowMixin`, the temporal resolution is not included here - it belongs to the
    shared generation data source (`Generation.time_resolution_minutes`), not to an individual
    window over it.
    """

    interval_start_minutes: int = Field(
        ...,
        description="Data interval starts at `t0 + interval_start_minutes`",
    )

    interval_end_minutes: int = Field(
        ...,
        description="Data interval ends at `t0 + interval_end_minutes`",
    )

    @model_validator(mode="after")
    def validate_interval_order(self) -> "GenerationWindow":
        """Validator for time interval fields."""
        start = self.interval_start_minutes
        end = self.interval_end_minutes
        if start > end:
            raise ValueError(
                f"interval_start_minutes ({start}) must be <= interval_end_minutes ({end})",
            )
        return self


class GenerationInputWindow(GenerationWindow, DropoutMixin):
    """Generation input window configuration model, used for `Generation.input`.

    Extends `GenerationWindow` with dropout configuration, since only the input window (not the
    prediction target) should ever be randomly masked out.
    """


class GenerationTargetWindow(GenerationWindow, FillValueMixin):
    """Generation target window configuration model, used for `Generation.target`."""


class Generation(Base):
    """Generation configuration model.

    Bundles the shared generation data source (`zarr_path`, `time_resolution_minutes`) with its
    `input` and `target` windows - two independently configurable time windows over the same
    underlying data. `time_resolution_minutes` describes generation's own native data cadence
    (used for gap detection and windowed slicing of generation's own data) - it is independent
    of `SamplingGrid.t0_resolution_minutes`, which is the cadence t0 candidates are enumerated
    at and may legitimately differ (e.g. generation stored every 5 minutes, sampled every 30).
    """

    zarr_path: str = Field(
        ...,
        description="Absolute or relative zarr filepath. Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    time_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="The temporal resolution of the generation data in minutes",
    )

    input: GenerationInputWindow | None = None
    target: GenerationTargetWindow | None = None

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
    """Configuration for the (location, time) grid that t0 times are sampled from.

    `locations_zarr_path` points to the locations metadata (location IDs and their
    coordinates) - see `ocf_data_sampler.load.locations.open_locations`.
    `t0_resolution_minutes` is the cadence t0 candidates are enumerated at, needed to compute
    valid t0 times regardless of which other input sources are configured - it is not any one
    source's own native data resolution (see `Generation.time_resolution_minutes` for that).
    """

    locations_zarr_path: str = Field(
        ...,
        description="Absolute or relative zarr filepath to the locations metadata. Prefix with "
        "a protocol like s3:// to read from alternative filesystems.",
    )

    t0_resolution_minutes: int = Field(
        ...,
        gt=0,
        description="The resolution of the t0 sampling grid, in minutes.",
    )


class SolarPosition(TimeWindowMixin):
    """Solar position configuration model."""


class DatetimeEncoding(TimeWindowMixin):
    """Datetime encoding configuration model."""


_embedding_type = list[tuple[str, Literal["cyclic", "linear"]]]
class T0Embedding(Base):
    """Configuration for the t0 time embedding."""

    embeddings: _embedding_type = Field(
        ...,
        description="""The periods to encode (e.g., "1h", "Nh", "1y", "Ny") and their representation
            (either "cyclic" or "linear"). When cyclic, the period is sin-cos embedded, else it is
            0-1 scaled as fraction through the period. Note that using "cyclic" adds 2 elements to
            the output vector to embed a period whilst "linear" adds only 1 element.""",
    )

    @field_validator("embeddings")
    def validate_embeddings(cls, embeddings: _embedding_type) -> _embedding_type:
        """Validate 'periods'."""
        for period, embedding_type in embeddings:

            if not isinstance(period, str):
                raise ValueError(f"Each period must be a string, found {type(period)}")

            unit = period[-1]
            if unit not in ["h", "y"]:
                raise ValueError(f"""Unit {unit} needs to in ["h","y"]""")

            if not period[:-1].isdigit():
                raise ValueError(f"{period[:-1]} not recognised as an integer")

            if unit=="y" and not int(period[:-1])>0:
                raise ValueError(f"When using unit y the period (={period[:-1]}) must be > 0")

            if unit=="h" and not (1<=int(period[:-1])<=24):
                raise ValueError(
                    f"When using unit h the period (={period[:-1]}) must be in interval [1, 24]",
                )

            if embedding_type not in ["cyclic", "linear"]:
                raise ValueError(f"Embedding ({embedding_type}) must be cyclic or linear")

        return embeddings


class PVNetDataConfig(Base):
    """Configuration model for the PVNet dataset."""

    sampling_grid: SamplingGrid
    satellite: Satellite | None = None
    nwp: MultiNWP | None = None
    generation: Generation | None = None
    solar_position: SolarPosition | None = None
    datetime_encoding: DatetimeEncoding | None = None
    t0_embedding: T0Embedding | None = None
