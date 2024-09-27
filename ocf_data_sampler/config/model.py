""" Configuration model for the dataset.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.

"""

import logging
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, RootModel, field_validator, ValidationInfo

from ocf_datapipes.utils.consts import NWP_PROVIDERS

logger = logging.getLogger(__name__)

providers = ["pvoutput.org", "solar_sheffield_passiv"]


class Base(BaseModel):
    """Pydantic Base model where no extras can be added"""

    class Config:
        """config class"""

        extra = "forbid"  # forbid use of extra kwargs


class General(Base):
    """General pydantic model"""

    name: str = Field("example", description="The name of this configuration file.")
    description: str = Field(
        "example configuration", description="Description of this configuration file"
    )


class DataSourceMixin(Base):
    """Mixin class, to add forecast and history minutes"""

    forecast_minutes: int = Field(
        None,
        ge=0,
        description="how many minutes to forecast in the future. "
        "If set to None, the value is defaulted to InputData.default_forecast_minutes",
    )
    history_minutes: int = Field(
        None,
        ge=0,
        description="how many historic minutes to use. "
        "If set to None, the value is defaulted to InputData.default_history_minutes",
    )


# noinspection PyMethodParameters
class DropoutMixin(Base):
    """Mixin class, to add dropout minutes"""

    dropout_timedeltas_minutes: Optional[List[int]] = Field(
        default=None,
        description="List of possible minutes before t0 where data availability may start. Must be "
        "negative or zero.",
    )

    dropout_fraction: float = Field(0, description="Chance of dropout being applied to each sample")

    @field_validator("dropout_timedeltas_minutes")
    def dropout_timedeltas_minutes_negative(cls, v: List[int]) -> List[int]:
        """Validate 'dropout_timedeltas_minutes'"""
        if v is not None:
            for m in v:
                assert m <= 0
        return v

    @field_validator("dropout_fraction")
    def dropout_fraction_valid(cls, v: float) -> float:
        """Validate 'dropout_fraction'"""
        assert 0 <= v <= 1
        return v


# noinspection PyMethodParameters
class TimeResolutionMixin(Base):
    """Time resolution mix in"""

    time_resolution_minutes: int = Field(
        ...,
        description="The temporal resolution of the data in minutes",
    )


class Satellite(DataSourceMixin, TimeResolutionMixin, DropoutMixin):
    """Satellite configuration model"""

    # Todo: remove 'satellite' from names
    satellite_zarr_path: Union[str, tuple[str], list[str]] = Field(
        ...,
        description="The path or list of paths which hold the satellite zarr",
    )
    satellite_channels: tuple = Field(
        ..., description="the satellite channels that are used"
    )
    satellite_image_size_pixels_height: int = Field(
        ...,
        description="The number of pixels of the height of the region of interest"
        " for non-HRV satellite channels.",
    )

    satellite_image_size_pixels_width: int = Field(
        ...,
        description="The number of pixels of the width of the region "
        "of interest for non-HRV satellite channels.",
    )

    live_delay_minutes: int = Field(
        30, description="The expected delay in minutes of the satellite data"
    )


# noinspection PyMethodParameters
class NWP(DataSourceMixin, TimeResolutionMixin, DropoutMixin):
    """NWP configuration model"""

    nwp_zarr_path: Union[str, tuple[str], list[str]] = Field(
        ...,
        description="The path which holds the NWP zarr",
    )
    nwp_channels: tuple = Field(
        ..., description="the channels used in the nwp data"
    )
    nwp_accum_channels: tuple = Field([], description="the nwp channels which need to be diffed")
    nwp_image_size_pixels_height: int = Field(..., description="The size of NWP spacial crop in pixels")
    nwp_image_size_pixels_width: int = Field(..., description="The size of NWP spacial crop in pixels")

    nwp_provider: str = Field(..., description="The provider of the NWP data")

    max_staleness_minutes: Optional[int] = Field(
        None,
        description="Sets a limit on how stale an NWP init time is allowed to be whilst still being"
        " used to construct an example. If set to None, then the max staleness is set according to"
        " the maximum forecast horizon of the NWP and the requested forecast length.",
    )


    @field_validator("nwp_provider")
    def validate_nwp_provider(cls, v: str) -> str:
        """Validate 'nwp_provider'"""
        if v.lower() not in NWP_PROVIDERS:
            message = f"NWP provider {v} is not in {NWP_PROVIDERS}"
            logger.warning(message)
            assert Exception(message)
        return v

    # Todo: put into time mixin when moving intervals there
    @field_validator("forecast_minutes")
    def forecast_minutes_divide_by_time_resolution(cls, v: int, info: ValidationInfo) -> int:
        if v % info.data["time_resolution_minutes"] != 0:
            message = "Forecast duration must be divisible by time resolution"
            logger.error(message)
            raise Exception(message)
        return v

    @field_validator("history_minutes")
    def history_minutes_divide_by_time_resolution(cls, v: int, info: ValidationInfo) -> int:
        if v % info.data["time_resolution_minutes"] != 0:
            message = "History duration must be divisible by time resolution"
            logger.error(message)
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


# noinspection PyMethodParameters
class GSP(DataSourceMixin, TimeResolutionMixin, DropoutMixin):
    """GSP configuration model"""

    gsp_zarr_path: str = Field(..., description="The path which holds the GSP zarr")
    gsp_image_size_pixels_height: int = Field(64, description="The size of GSP spacial crop in pixels")
    gsp_image_size_pixels_width: int = Field(64, description="The size of GSP spacial crop in pixels")

    # Todo: needs to be changes from hardcode when moving to mixin
    @field_validator("history_minutes")
    def history_minutes_divide_by_30(cls, v):
        """Validate 'history_minutes'"""
        assert v % 30 == 0  # this means it also divides by 5
        return v

    @field_validator("forecast_minutes")
    def forecast_minutes_divide_by_30(cls, v):
        """Validate 'forecast_minutes'"""
        assert v % 30 == 0  # this means it also divides by 5
        return v


# noinspection PyPep8Naming
class InputData(Base):
    """
    Input data model.
    """

    satellite: Optional[Satellite] = None
    nwp: Optional[MultiNWP] = None
    gsp: Optional[GSP] = None


class Configuration(Base):
    """Configuration model for the dataset"""

    general: General = General()
    input_data: InputData = InputData()