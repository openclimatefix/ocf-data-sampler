"""Configuration model for the dataset.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.

Example:

    from ocf_data_sampler.config import Configuration
    config = Configuration(**config_dict)
"""

import logging
from typing import Dict, List, Optional
from typing_extensions import Self

from pydantic import BaseModel, Field, RootModel, field_validator, ValidationInfo, model_validator

from ocf_data_sampler.constants import NWP_PROVIDERS

logger = logging.getLogger(__name__)

providers = ["pvoutput.org", "solar_sheffield_passiv"]


class Base(BaseModel):
    """Pydantic Base model where no extras can be added"""

    class Config:
        """config class"""

        extra = "forbid"  # forbid use of extra kwargs


class General(Base):
    """General pydantic model"""

    name: str = Field("example", description="The name of this configuration file")
    description: str = Field(
        "example configuration", description="Description of this configuration file"
    )


class TimeWindowMixin(Base):
    """Mixin class, to add interval start, end and resolution minutes"""

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
    
    @model_validator(mode='after')
    def check_interval_range(cls, values):
        if values.interval_start_minutes > values.interval_end_minutes:
            raise ValueError('interval_start_minutes must be <= interval_end_minutes')
        return values

    @field_validator("interval_start_minutes")
    def interval_start_minutes_divide_by_time_resolution(cls, v: int, info: ValidationInfo) -> int:
        if v % info.data["time_resolution_minutes"] != 0:
            raise ValueError("interval_start_minutes must be divisible by time_resolution_minutes")
        return v

    @field_validator("interval_end_minutes")
    def interval_end_minutes_divide_by_time_resolution(cls, v: int, info: ValidationInfo) -> int:
        if v % info.data["time_resolution_minutes"] != 0:
            raise ValueError("interval_end_minutes must be divisible by time_resolution_minutes")
        return v



# noinspection PyMethodParameters
class DropoutMixin(Base):
    """Mixin class, to add dropout minutes"""

    dropout_timedeltas_minutes: Optional[List[int]] = Field(
        default=None,
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
    def dropout_timedeltas_minutes_negative(cls, v: List[int]) -> List[int]:
        """Validate 'dropout_timedeltas_minutes'"""
        if v is not None:
            for m in v:
                assert m <= 0, "Dropout timedeltas must be negative"
        return v

    @model_validator(mode="after")
    def dropout_instructions_consistent(self) -> Self:
        if self.dropout_fraction == 0:
            if self.dropout_timedeltas_minutes is not None:
                raise ValueError("To use dropout timedeltas dropout fraction should be > 0")
        else:
            if self.dropout_timedeltas_minutes is None:
                raise ValueError("To dropout fraction > 0 requires a list of dropout timedeltas")
        return self


class SpatialWindowMixin(Base):
    """Mixin class, to add path and image size"""

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
    """Satellite configuration model"""
    
    zarr_path: str | tuple[str] | list[str] = Field(
        ...,
        description="The path or list of paths which hold the data zarr",
    )

    channels: list[str] = Field(
        ..., description="the satellite channels that are used"
    )


# noinspection PyMethodParameters
class NWP(TimeWindowMixin, DropoutMixin, SpatialWindowMixin):
    """NWP configuration model"""
    
    zarr_path: str | tuple[str] | list[str] = Field(
        ...,
        description="The path or list of paths which hold the data zarr",
    )
    
    channels: list[str] = Field(
        ..., description="the channels used in the nwp data"
    )

    provider: str = Field(..., description="The provider of the NWP data")

    accum_channels: list[str] = Field([], description="the nwp channels which need to be diffed")

    max_staleness_minutes: Optional[int] = Field(
        None,
        description="Sets a limit on how stale an NWP init time is allowed to be whilst still being"
        " used to construct an example. If set to None, then the max staleness is set according to"
        " the maximum forecast horizon of the NWP and the requested forecast length.",
    )


    @field_validator("provider")
    def validate_provider(cls, v: str) -> str:
        """Validate 'provider'"""
        if v.lower() not in NWP_PROVIDERS:
            message = f"NWP provider {v} is not in {NWP_PROVIDERS}"
            logger.warning(message)
            raise Exception(message)
        return v


class MultiNWP(RootModel):
    """Configuration for multiple NWPs"""

    root: Dict[str, NWP]

    def __getattr__(self, item):
        return self.root[item]

    def __getitem__(self, item):
        return self.root[item]

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        """Returns dictionary-like keys"""
        return self.root.keys()

    def items(self):
        """Returns dictionary-like items"""
        return self.root.items()


class GSP(TimeWindowMixin, DropoutMixin):
    """GSP configuration model"""

    zarr_path: str = Field(..., description="The path which holds the GSP zarr")


class Site(TimeWindowMixin, DropoutMixin):
    """Site configuration model"""

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



# noinspection PyPep8Naming
class InputData(Base):
    """Input data model"""

    satellite: Optional[Satellite] = None
    nwp: Optional[MultiNWP] = None
    gsp: Optional[GSP] = None
    site: Optional[Site] = None


class Configuration(Base):
    """Configuration model for the dataset"""

    general: General = General()
    input_data: InputData = InputData()
