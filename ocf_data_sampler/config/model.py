"""Configuration model for the dataset.

Absolute or relative zarr filepath(s).
Prefix with a protocol like s3:// to read from alternative filesystems.
"""

from collections.abc import Iterator

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from typing_extensions import override

from ocf_data_sampler.constants import NWP_PROVIDERS


class Base(BaseModel):
    """Pydantic Base model where no extras can be added."""

    class Config:
        """Config class."""

        extra = "forbid"  # forbid use of extra kwargs


class General(Base):
    """General pydantic model."""

    name: str = Field("example", description="The name of this configuration file")
    description: str = Field(
        "example configuration",
        description="Description of this configuration file",
    )


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


class DropoutMixin(Base):
    """Mixin class, to add dropout minutes."""

    dropout_timedeltas_minutes: list[int] = Field(
        default=[],
        description="List of possible minutes before t0 where data availability may start. Must be "
        "negative or zero.",
    )

    dropout_fraction: float = Field(
        default=0,
        description="Chance of dropout being applied to each sample",
        ge=0,
        le=1,
    )

    @field_validator("dropout_timedeltas_minutes")
    def dropout_timedeltas_minutes_negative(cls, v: list[int]) -> list[int]:
        """Validate 'dropout_timedeltas_minutes'."""
        for m in v:
            if m > 0:
                raise ValueError("Dropout timedeltas must be negative")
        return v

    @model_validator(mode="after")
    def dropout_instructions_consistent(self) -> "DropoutMixin":
        """Validator for dropout instructions."""
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
        ge=0,
        description="The number of pixels of the height of the region of interest",
    )

    image_size_pixels_width: int = Field(
        ...,
        ge=0,
        description="The number of pixels of the width of the region of interest",
    )


class Satellite(TimeWindowMixin, DropoutMixin, SpatialWindowMixin):
    """Satellite configuration model."""

    zarr_path: str | tuple[str] | list[str] = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[str] = Field(
        ...,
        description="the satellite channels that are used",
    )


class NWP(TimeWindowMixin, DropoutMixin, SpatialWindowMixin):
    """NWP configuration model."""

    zarr_path: str | tuple[str] | list[str] = Field(
        ...,
        description="Absolute or relative zarr filepath(s). Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    channels: list[str] = Field(
        ...,
        description="the channels used in the nwp data",
    )

    provider: str = Field(..., description="The provider of the NWP data")

    accum_channels: list[str] = Field([], description="the nwp channels which need to be diffed")

    max_staleness_minutes: int | None = Field(
        None,
        description="Sets a limit on how stale an NWP init time is allowed to be whilst still being"
        " used to construct an example. If set to None, then the max staleness is set according to"
        " the maximum forecast horizon of the NWP and the requested forecast length.",
    )

    @field_validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validator for 'provider'."""
        if v.lower() not in NWP_PROVIDERS:
            raise OSError(f"NWP provider {v} is not in {NWP_PROVIDERS}")
        return v


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


class GSP(TimeWindowMixin, DropoutMixin):
    """GSP configuration model."""

    zarr_path: str = Field(
        ...,
        description="Absolute or relative zarr filepath. Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )


class Site(TimeWindowMixin, DropoutMixin):
    """Site configuration model."""

    file_path: str = Field(
        ...,
        description="The NetCDF files holding the power timeseries.",
    )
    metadata_file_path: str = Field(
        ...,
        description="The CSV files describing power system",
    )

    # TODO validate the netcdf for sites
    # TODO validate the csv for metadata


class SolarPosition(TimeWindowMixin):
    """Solar position configuration model."""


class InputData(Base):
    """Input data model."""

    satellite: Satellite | None = None
    nwp: MultiNWP | None = None
    gsp: GSP | None = None
    site: Site | None = None
    solar_position: SolarPosition | None = None


class Configuration(Base):
    """Configuration model for the dataset."""

    general: General = General()
    input_data: InputData = InputData()
