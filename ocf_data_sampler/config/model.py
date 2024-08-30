""" Configuration model for the dataset.

All paths must include the protocol prefix.  For local files,
it's sufficient to just start with a '/'.  For aws, start with 's3://',
for gcp start with 'gs://'.

"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

import git
import numpy as np
from pathy import Pathy
from pydantic import BaseModel, Field, RootModel, model_validator, field_validator, ValidationInfo

from ocf_datapipes.utils.consts import (
    NWP_PROVIDERS,
    NWP_VARIABLE_NAMES,
    RSS_VARIABLE_NAMES,
)

IMAGE_SIZE_PIXELS = 64
IMAGE_SIZE_PIXELS_FIELD = Field(
    IMAGE_SIZE_PIXELS, description="The number of pixels of the region of interest."
)
METERS_PER_PIXEL_FIELD = Field(2000, description="The number of meters per pixel.")
METERS_PER_ROI = Field(128_000, description="The number of meters of region of interest.")

DEFAULT_N_GSP_PER_EXAMPLE = 32
DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE = 2048

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


class Git(Base):
    """Git model"""

    hash: str = Field(
        ..., description="The git hash of nowcasting_dataset when a dataset is created."
    )
    message: str = Field(..., description="The git message for when a dataset is created.")
    committed_date: datetime = Field(
        ..., description="The git datestamp for when a dataset is created."
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

    log_level: str = Field(
        "DEBUG",
        description="The logging level for this data source. "
        "This is the default value and can be set in each data source",
    )

    # Todo: are these still needed?
    @property
    def seq_length_30_minutes(self):
        """How many steps are there in 30 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 30 + 1))

    @property
    def seq_length_5_minutes(self):
        """How many steps are there in 5 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 5 + 1))

    @property
    def seq_length_60_minutes(self):
        """How many steps are there in 60 minute datasets"""
        return int(np.ceil((self.history_minutes + self.forecast_minutes) / 60 + 1))

    @property
    def history_seq_length_5_minutes(self):
        """How many historical steps are there in 5 minute datasets"""
        return int(np.ceil(self.history_minutes / 5))

    @property
    def history_seq_length_30_minutes(self):
        """How many historical steps are there in 30 minute datasets"""
        return int(np.ceil(self.history_minutes / 30))

    @property
    def history_seq_length_60_minutes(self):
        """How many historical steps are there in 60 minute datasets"""
        return int(np.ceil(self.history_minutes / 60))


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


# Todo: do we ever use this?
# noinspection PyMethodParameters
class SystemDropoutMixin(Base):
    """Mixin class, to add independent system dropout"""

    system_dropout_timedeltas_minutes: Optional[List[int]] = Field(
        None,
        description="List of possible minutes before t0 where data availability may start. Must be "
        "negative or zero. Each system in a sample is delayed independently from the other by "
        "values randomly selected from this list.",
    )

    # The degree of system dropout for each returned sample will be randomly drawn from
    # the range [system_dropout_fraction_min, system_dropout_fraction_max]
    system_dropout_fraction_min: float = Field(0, description="Min chance of system dropout")
    system_dropout_fraction_max: float = Field(0, description="Max chance of system dropout")

    @field_validator("system_dropout_fraction_min", "system_dropout_fraction_max")
    def validate_system_dropout_fractions(cls, v: float):
        """Validate dropout fraction values"""
        assert 0 <= v <= 1
        return v

    @model_validator(mode="after")
    def validate_system_dropout_fraction_range(self):
        """Ensure positive dropout fraction range"""
        assert self.system_dropout_fraction_min <= self.system_dropout_fraction_max
        return self


# noinspection PyMethodParameters
class TimeResolutionMixin(Base):
    """Time resolution mix in"""

    time_resolution_minutes: int = Field(
        5,
        description="The temporal resolution (in minutes) of the satellite images."
        "Note that this needs to be divisible by 5.",
    )

    @field_validator("time_resolution_minutes")
    def forecast_minutes_divide_by_5(cls, v: int) -> int:
        """Validate 'forecast_minutes'"""
        assert v % 5 == 0, f"The time resolution ({v}) is not divisible by 5"
        return v


class XYDimensionalNames(Base):
    """X and Y dimensions names"""

    x_dim_name: str = Field(
        "x_osgb",
        description="The x dimension name. Should be either x_osgb or longitude",
    )

    y_dim_name: str = Field(
        "y_osgb",
        description="The y dimension name. Should be either y_osgb or latitude",
    )

    @model_validator(mode="after")
    def check_x_y_dimension_names(self):
        """Check that the x and y dimension pair up correctly"""

        x_dim_name = self.x_dim_name
        y_dim_name = self.y_dim_name

        assert x_dim_name in ["x_osgb", "longitude", "x"]
        assert y_dim_name in ["y_osgb", "latitude", "y"]

        if x_dim_name == "x":
            assert y_dim_name == "y"

        if x_dim_name == "x_osgb":
            assert y_dim_name == "y_osgb"

        if x_dim_name == "longitude":
            assert y_dim_name == "latitude"

        return self


# noinspection PyMethodParameters
class PVFiles(BaseModel):
    """Model to hold pv file and metadata file"""

    pv_filename: Optional[str] = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_timeseries_batch.nc",
        description="The NetCDF files holding the solar PV power timeseries.",
    )
    pv_metadata_filename: Optional[str] = Field(
        "gs://solar-pv-nowcasting-data/PV/PVOutput.org/UK_PV_metadata.csv",
        description="The CSV files describing each PV system.",
    )
    inferred_metadata_filename: Optional[str] = Field(
        None,
        description="The CSV files describing inferred PV metadata for each system.",
    )

    label: Optional[str] = Field(providers[0], description="Label of where the pv data came from")

    @field_validator("label")
    def v_label0(cls, v: str) -> str:
        """Validate 'label'"""
        if v not in providers:
            message = f"provider {v} not in {providers}"
            logger.error(message)
            raise Exception(message)
        return v


# noinspection PyMethodParameters
class PV(
    DataSourceMixin, TimeResolutionMixin, XYDimensionalNames, DropoutMixin, SystemDropoutMixin
):
    """PV configuration model"""

    pv_files_groups: List[PVFiles] = [PVFiles()]

    n_pv_systems_per_example: int = Field(
        DEFAULT_N_PV_SYSTEMS_PER_EXAMPLE,
        description="The number of PV systems samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    pv_image_size_meters_height: int = METERS_PER_ROI
    pv_image_size_meters_width: int = METERS_PER_ROI

    pv_filename: Optional[str] = Field(
        None,
        description="The NetCDF files holding the solar PV power timeseries.",
    )
    pv_metadata_filename: Optional[str] = Field(
        None,
        description="The CSV files describing each PV system.",
    )

    pv_ml_ids: List[int] = Field(
        None,
        description="List of the ML IDs of the PV systems you'd like to filter to.",
    )

    is_live: bool = Field(
        False, description="Option if to use live data from the nowcasting pv database"
    )

    live_interpolate_minutes: int = Field(
        30, description="The number of minutes we allow PV data to interpolate"
    )
    live_load_extra_minutes: int = Field(
        0,
        description="The number of extra minutes in the past we should load. Then the recent "
        "values can be interpolated, and the extra minutes removed. This is "
        "because some live data takes ~1 hour to come in.",
    )

    time_resolution_minutes: int = Field(
        5,
        description="The temporal resolution (in minutes) of the data."
        "Note that this needs to be divisible by 5.",
    )

    # Todo: is this still relevant?
    @classmethod
    def model_validation(cls, v):
        """Move old way of storing filenames to new way"""

        if (v.pv_filename is not None) and (v.pv_metadata_filename is not None):
            logger.warning(
                "Loading pv files the old way, and moving them the new way. "
                "Please update configuration file"
            )
            label = (
                "pv_output.org" if "pvoutput" in v.pv_filename.lower() else "solar_sheffield_passiv"
            )
            pv_file = PVFiles(
                pv_filename=v.pv_filename, pv_metadata_filename=v.pv_metadata_filename, label=label
            )
            v.pv_files_groups = [pv_file]
            v.pv_filename = None
            v.pv_metadata_filename = None

        return v
    
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


class Satellite(DataSourceMixin, TimeResolutionMixin, DropoutMixin):
    """Satellite configuration model"""

    satellite_zarr_path: Union[str, tuple[str], list[str]] = Field(
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/OSGB36/all_zarr_int16_single_timestep.zarr",  # noqa: E501
        description="The path or list of paths which hold the satellite zarr.",
    )
    satellite_channels: tuple = Field(
        RSS_VARIABLE_NAMES[1:], description="the satellite channels that are used"
    )
    satellite_image_size_pixels_height: int = Field(
        IMAGE_SIZE_PIXELS_FIELD.default // 3,
        description="The number of pixels of the height of the region of interest"
        " for non-HRV satellite channels.",
    )
    satellite_image_size_pixels_width: int = Field(
        IMAGE_SIZE_PIXELS_FIELD.default // 3,
        description="The number of pixels of the width of the region "
        "of interest for non-HRV satellite channels.",
    )
    satellite_meters_per_pixel: int = Field(
        METERS_PER_PIXEL_FIELD.default * 3,
        description="The number of meters per pixel for non-HRV satellite channels.",
    )

    is_live: bool = Field(
        False,
        description="Option to use live data from the satellite consumer. "
        "This is useful because the data is about ~30 minutes behind, "
        "so we need to expect that",
    )

    live_delay_minutes: int = Field(
        30, description="The expected delay in minutes of the satellite data"
    )

    time_resolution_minutes: int = Field(
        5,
        description="The temporal resolution (in minutes) of the data."
        "Note that this needs to be divisible by 5.",
    )


# noinspection PyMethodParameters
class NWP(DataSourceMixin, TimeResolutionMixin, XYDimensionalNames, DropoutMixin):
    """NWP configuration model"""

    nwp_zarr_path: Union[str, tuple[str], list[str]] = Field(
        "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV__2018-01_to_2019-12__chunks__variable10__init_time1__step1__x548__y704__.zarr",  # noqa: E501
        description="The path which holds the NWP zarr.",
    )
    nwp_channels: tuple = Field(
        NWP_VARIABLE_NAMES["ukv"], description="the channels used in the nwp data"
    )
    nwp_accum_channels: tuple = Field([], description="the nwp channels which need to be diffed")
    nwp_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    nwp_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    nwp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD
    nwp_provider: str = Field("ukv", description="The provider of the NWP data")
    index_by_id: bool = Field(
        False, description="If the NWP data has an id coordinate, not x and y."
    )

    max_staleness_minutes: Optional[int] = Field(
        None,
        description="Sets a limit on how stale an NWP init time is allowed to be whilst still being"
        " used to construct an example. If set to None, then the max staleness is set according to"
        " the maximum forecast horizon of the NWP and the requested forecast length.",
    )

    coarsen_to_degrees: Optional[float] = Field(
        0.1, description="The number of degrees to coarsen the NWP data to"
    )

    @field_validator("nwp_provider")
    def validate_nwp_provider(cls, v: str) -> str:
        """Validate 'nwp_provider'"""
        if v.lower() not in NWP_PROVIDERS:
            message = f"NWP provider {v} is not in {NWP_PROVIDERS}"
            logger.warning(message)
            assert Exception(message)
        return v
    
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

    gsp_zarr_path: str = Field("gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr")
    n_gsp_per_example: int = Field(
        DEFAULT_N_GSP_PER_EXAMPLE,
        description="The number of GSP samples per example. "
        "If there are less in the ROI then the data is padded with zeros. ",
    )
    gsp_image_size_pixels_height: int = IMAGE_SIZE_PIXELS_FIELD
    gsp_image_size_pixels_width: int = IMAGE_SIZE_PIXELS_FIELD
    gsp_meters_per_pixel: int = METERS_PER_PIXEL_FIELD
    metadata_only: bool = Field(False, description="Option to only load metadata.")

    is_live: bool = Field(
        False, description="Option if to use live data from the nowcasting GSP/Forecast database"
    )

    live_interpolate_minutes: int = Field(
        60, description="The number of minutes we allow GSP data to be interpolated"
    )
    live_load_extra_minutes: int = Field(
        60,
        description="The number of extra minutes in the past we should load. Then the recent "
        "values can be interpolated, and the extra minutes removed. This is "
        "because some live data takes ~1 hour to come in.",
    )
    time_resolution_minutes: int = Field(
        30,
        description="The temporal resolution (in minutes) of the data."
        "Note that this needs to be divisible by 5.",
    )

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

    pv: Optional[PV] = None
    satellite: Optional[Satellite] = None
    nwp: Optional[MultiNWP] = None
    gsp: Optional[GSP] = None

    default_forecast_minutes: int = Field(
        60,
        ge=0,
        description="how many minutes to forecast in the future. "
        "This sets the default for all the data sources if they are not set.",
    )
    default_history_minutes: int = Field(
        30,
        ge=0,
        description="how many historic minutes are used. "
        "This sets the default for all the data sources if they are not set.",
    )

    @property
    def default_seq_length_5_minutes(self):
        """How many steps are there in 5 minute datasets"""
        return int((self.default_history_minutes + self.default_forecast_minutes) / 5 + 1)

    @model_validator(mode="after")
    def set_forecast_and_history_minutes(self):
        """
        Set default history and forecast values, if needed.

        Run through the different data sources and  if the forecast or history minutes are not set,
        then set them to the default values
        """

        ALL_DATA_SOURCE_NAMES = (
            "pv",
            "satellite",
            # "nwp", # nwp is treated separately
            "gsp",
        )
        enabled_data_sources = [
            data_source_name
            for data_source_name in ALL_DATA_SOURCE_NAMES
            if getattr(self, data_source_name) is not None
        ]

        for data_source_name in enabled_data_sources:
            if getattr(self, data_source_name).forecast_minutes is None:
                getattr(self, data_source_name).forecast_minutes = self.default_forecast_minutes

            if getattr(self, data_source_name).history_minutes is None:
                getattr(self, data_source_name).history_minutes = self.default_history_minutes

        if self.nwp is not None:
            for k in self.nwp.keys():
                if self.nwp[k].forecast_minutes is None:
                    self.nwp[k].forecast_minutes = self.default_forecast_minutes

                if self.nwp[k].history_minutes is None:
                    self.nwp[k].history_minutes = self.default_history_minutes

        return self

    @classmethod
    def set_all_to_defaults(cls):
        """Returns an InputData instance with all fields set to their default values.

        Used for unittests.
        """
        return cls(
            pv=PV(),
            satellite=Satellite(),
            nwp=dict(UKV=NWP()),
            gsp=GSP(),
        )


class Configuration(Base):
    """Configuration model for the dataset"""

    general: General = General()
    input_data: InputData = InputData()
    git: Optional[Git] = None

    def set_base_path(self, base_path: str):
        """Append base_path to all paths. Mostly used for testing."""
        base_path = Pathy(base_path)
        path_attrs = [
            "pv.pv_filename",
            "pv.pv_metadata_filename",
            "satellite.satellite_zarr_path",
            "nwp.nwp_zarr_path",
            "gsp.gsp_zarr_path",
        ]
        for cls_and_attr_name in path_attrs:
            cls_name, attribute = cls_and_attr_name.split(".")
            cls = getattr(self.input_data, cls_name)
            path = getattr(getattr(self.input_data, cls_name), attribute)
            path = base_path / path
            setattr(cls, attribute, path)
            setattr(self.input_data, cls_name, cls)


def set_git_commit(configuration: Configuration):
    """
    Set the git information in the configuration file

    Args:
        configuration: configuration object

    Returns: configuration object with git information

    """

    repo = git.Repo(search_parent_directories=True)
    git.refresh("/usr/bin/git")

    git_details = Git(
        hash=repo.head.object.hexsha,
        committed_date=datetime.fromtimestamp(repo.head.object.committed_date),
        message=repo.head.object.message,
    )

    configuration.git = git_details

    return configuration
