"""Functions to convert xarray datasets to numpy samples."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.indexing import get_string_index
from ocf_data_sampler.common.time_utils import (
    date_range,
    datetime_floor,
    get_posix_timestamp,
    minutes,
)
from ocf_data_sampler.common.types import DataArrayLike
from ocf_data_sampler.config.model import PVNetDataConfig, TimeWindowMixin
from ocf_data_sampler.datasets.pvnet.types import Location, NumpySample, SourceDict
from ocf_data_sampler.transforms.solar import calculate_azimuth_and_elevation
from ocf_data_sampler.transforms.time_encodings import build_t0_embedding, encode_datetimes


def build_numpy_sample[TArray: DataArrayLike](
    source_data: SourceDict[TArray],
    t0: np.datetime64,
    location: Location,
    config: PVNetDataConfig,
    include_extra_metadata: bool = False,
) -> NumpySample:
    """Build source arrays, metadata and configured features for one time and location.

    Source arrays must already be preprocessed by `preprocess_source_data`.

    Field shapes, units and optionality are documented in `NumpySample`.
    Arrays may share memory with the prepared input.

    Args:
        source_data: Preprocessed source arrays for one location and forecast time.
        t0: Forecast reference time.
        location: location of the sample
        config: PVNetDataConfig object
        include_extra_metadata: Whether to add additional non-essential metadata to the sample

    Returns:
        Source data arrays, metadata and configured features keyed by sample field name.
    """
    sample: NumpySample = {
        "location_id": location.id,
        "t0": get_posix_timestamp(t0),
    }

    add_source_arrays(sample, source_data)

    if include_extra_metadata:
        sample["metadata"] = {
            "location_longitude": location.longitude,
            "location_latitude": location.latitude,
        }

        add_source_metadata(sample["metadata"], source_data)

    if (t0_config := config.t0_embedding) is not None:
        sample["t0_embedding"] = build_t0_embedding(t0, t0_config.embeddings)

    if (datetime_config := config.datetime_encoding) is not None:
        datetimes = _build_feature_datetimes(t0, datetime_config)
        datetime_features = encode_datetimes(datetimes=datetimes)
        for key in ("date_sin", "date_cos", "time_sin", "time_cos"):
            sample[key] = datetime_features[key]

    if (solar_config := config.solar_position) is not None:
        datetimes = _build_feature_datetimes(t0, solar_config)
        _add_solar_features(sample, datetimes, lon=location.longitude, lat=location.latitude)

    return sample


def _build_feature_datetimes(
    t0: np.datetime64,
    config: TimeWindowMixin,
) -> NDArray[np.datetime64]:
    """Build an inclusive time range relative to t0 floored to the feature resolution."""
    time_resolution = minutes(config.time_resolution_minutes)
    aligned_t0 = datetime_floor(t0, time_resolution)
    return date_range(
        aligned_t0 + minutes(config.interval_start_minutes),
        aligned_t0 + minutes(config.interval_end_minutes),
        freq=time_resolution,
    )


def add_source_arrays[TArray: DataArrayLike](
    sample: NumpySample,
    source_data: SourceDict[TArray],
) -> None:
    """Add source data arrays to the sample in place.

    Generation timestamps are float64 POSIX seconds. Added arrays may share
    memory with the source arrays.

    Args:
        sample: Sample to populate.
        source_data: Preprocessed source arrays for one sample.
    """
    for key in ("generation_input", "generation_target"):
        if key not in source_data:
            continue

        da_gen = source_data[key]

        # Generation is already normalised by capacity; capacity remains in MW.
        gen_params = da_gen["gen_param"].values
        gen_idx = get_string_index(gen_params, "generation_mw")
        cap_idx = get_string_index(gen_params, "capacity_mwp")

        sample.update(
            {
                key: da_gen.isel(gen_param=gen_idx).values,
                f"{key}_capacity_mwp": da_gen.isel(gen_param=cap_idx).values,
                f"{key}_time_utc": get_posix_timestamp(da_gen["time_utc"].values),
            },
        )

    if "sat" in source_data:
        sample["satellite"] = source_data["sat"].values

    if "nwp" in source_data:
        sample["nwp"] = {
            nwp_name: da_nwp.values for nwp_name, da_nwp in source_data["nwp"].items()
        }


def add_source_metadata[TArray: DataArrayLike](
    metadata: dict[str, Any],
    source_data: SourceDict[TArray],
) -> None:
    """Add source diagnostics to an existing metadata dictionary in place.

    Absolute timestamps are float64 POSIX seconds; NWP lead times are float32 hours.
    Coordinate arrays retain source units and dtypes and may share source memory.

    Args:
        metadata: Metadata dictionary to populate, retaining unrelated entries.
        source_data: Source arrays for one sample.
    """
    if "sat" in source_data:
        da_sat = source_data["sat"]
        metadata.update({
            "satellite_time_utc": get_posix_timestamp(da_sat["time_utc"].values),
            "satellite_x_geostationary": da_sat["x_geostationary"].values,
            "satellite_y_geostationary": da_sat["y_geostationary"].values,
        })

    if "nwp" in source_data:
        metadata["nwp"] = {}
        for nwp_name, da_nwp in source_data["nwp"].items():
            step_hours = (da_nwp["step"].values / np.timedelta64(1, "h")).astype(np.float32)
            metadata["nwp"][nwp_name] = {
                "init_time_utc": get_posix_timestamp(da_nwp["init_time_utc"].values),
                "step_hours": step_hours,
            }


def _add_solar_features(
    sample: NumpySample,
    datetimes: NDArray[np.datetime64],
    lon: float,
    lat: float,
) -> None:
    """Add scaled float32 solar coordinates to the sample in-place.

    Args:
        sample: Sample to populate.
        datetimes: Datetimes for which to calculate the solar coordinates.
        lon: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        lat: Latitude in decimal degrees. Positive north of equator, negative to south.
    """
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Scale azimuth from [0, 360] and elevation from [-90, 90] degrees to [0, 1].
    azimuth = azimuth / 360
    elevation = elevation / 180 + 0.5

    sample["solar_azimuth"] = azimuth.astype(np.float32)
    sample["solar_elevation"] = elevation.astype(np.float32)
