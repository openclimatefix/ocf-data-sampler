"""Configuration model for the dataset.

Absolute or relative zarr filepath(s).
Prefix with a protocol like s3:// to read from alternative filesystems.
"""

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from typing_extensions import override

NWP_PROVIDERS = [
    "ukv",
    "ecmwf",
    "mo_global",
    "gfs",
    "icon_eu",
    "cloudcasting",
]


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

    dropout_fraction: float|list[float] = Field(
        default=0,
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
    def dropout_fractions(cls, dropout_frac: float|list[float]) -> float|list[float]:
        """Validate 'dropout_frac'."""
        from math import isclose
        if isinstance(dropout_frac, float):
            if not (dropout_frac <= 1):
                raise ValueError("Input should be less than or equal to 1")
            elif not (dropout_frac >= 0):
                raise ValueError("Input should be greater than or equal to 0")

        elif isinstance(dropout_frac, list):
            if not dropout_frac:
                raise ValueError("List cannot be empty")

            if not all(isinstance(i, float) for i in dropout_frac):
                raise ValueError("All elements in the list must be floats")

            if not all(0 <= i <= 1 for i in dropout_frac):
                raise ValueError("Each float in the list must be between 0 and 1")

            if not isclose(sum(dropout_frac), 1.0, rel_tol=1e-9):
                raise ValueError("Sum of all floats in the list must be 1.0")


        else:
            raise TypeError("Must be either a float or a list of floats")
        return dropout_frac


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


class NormalisationValues(Base):
    """Normalisation mean and standard deviation."""
    mean: float = Field(..., description="Mean value for normalization")
    std: float = Field(..., gt=0, description="Standard deviation (must be positive)")


class NormalisationConstantsMixin(Base):
    """Normalisation constants for multiple channels."""
    normalisation_constants: dict[str, NormalisationValues]

    @property
    def channel_means(self) -> dict[str, float]:
        """Return the channel means."""
        return {
            channel: norm_values.mean
            for channel, norm_values in self.normalisation_constants.items()
        }


    @property
    def channel_stds(self) -> dict[str, float]:
        """Return the channel standard deviations."""
        return {
            channel: norm_values.std
            for channel, norm_values in self.normalisation_constants.items()
        }


class Satellite(TimeWindowMixin, DropoutMixin, SpatialWindowMixin, NormalisationConstantsMixin):
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

    @model_validator(mode="after")
    def check_all_channel_have_normalisation_constants(self) -> "Satellite":
        """Check that all the channels have normalisation constants."""
        normalisation_channels = set(self.normalisation_constants.keys())
        missing_norm_values = set(self.channels) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalsation constants must be provided for all channels. Missing values for "
                f"channels: {missing_norm_values}",
            )
        return self


class NWP(TimeWindowMixin, DropoutMixin, SpatialWindowMixin, NormalisationConstantsMixin):
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
    public: bool = Field(False, description="Whether the NWP data is public or private")

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
        if v.lower() not in NWP_PROVIDERS:
            raise OSError(f"NWP provider {v} is not in {NWP_PROVIDERS}")
        return v


    @model_validator(mode="after")
    def check_all_channel_have_normalisation_constants(self) -> "NWP":
        """Check that all the channels have normalisation constants."""
        normalisation_channels = set(self.normalisation_constants.keys())
        non_accum_channels = [c for c in self.channels if c not in self.accum_channels]
        accum_channel_names = [f"diff_{c}" for c in self.accum_channels]

        missing_norm_values = set(non_accum_channels) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalsation constants must be provided for all channels. Missing values for "
                f"channels: {missing_norm_values}",
            )

        missing_norm_values = set(accum_channel_names) - set(normalisation_channels)
        if len(missing_norm_values)>0:
            raise ValueError(
                "Normalsation constants must be provided for all channels. Accumulated "
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


class GSP(TimeWindowMixin, DropoutMixin):
    """GSP configuration model."""

    zarr_path: str = Field(
        ...,
        description="Absolute or relative zarr filepath. Prefix with a protocol like s3:// "
        "to read from alternative filesystems.",
    )

    boundaries_version: Literal["20220314", "20250109"] = Field(
        "20220314",
        description="Version of the GSP boundaries to use. Options are '20220314' or '20250109'.",
    )

    public: bool = Field(False, description="Whether the NWP data is public or private")


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
