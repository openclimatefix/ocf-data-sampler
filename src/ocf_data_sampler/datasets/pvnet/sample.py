"""Functions to convert xarray datasets to numpy samples."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.time_utils import (
    date_range,
    datetime_floor,
    get_posix_timestamp,
    minutes,
)
from ocf_data_sampler.config.model import PVNetDataConfig, TimeWindowMixin
from ocf_data_sampler.datasets.pvnet.types import Location, NumpySample, SourceDict
from ocf_data_sampler.features.solar import calculate_azimuth_and_elevation
from ocf_data_sampler.features.time_encodings import encode_datetimes, encode_t0


def build_numpy_sample(
    dataset_dict: SourceDict,
    t0: np.datetime64,
    location: Location,
    config: PVNetDataConfig,
    include_extra_metadata: bool = False,
) -> NumpySample:
    """Build source arrays, metadata and configured features for one time and location.

    Source arrays must already be preprocessed by `preprocess_dataset_dict`.

    Field shapes, units and optionality are documented in `NumpySample`.
    Arrays may share memory with the prepared input.

    Args:
        dataset_dict: Preprocessed source arrays for one location and forecast time.
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

    add_source_arrays(sample, dataset_dict)

    if include_extra_metadata:
        sample["metadata"] = {
            "location_longitude": location.longitude,
            "location_latitude": location.latitude,
        }

        add_source_metadata(sample["metadata"], dataset_dict)

    if (t0_config := config.t0_embedding) is not None:
        sample["t0_embedding"] = encode_t0(t0, t0_config.embeddings)

    if (datetime_config := config.datetime_encoding) is not None:
        datetimes = _build_feature_datetimes(t0, datetime_config)
        sample.update(encode_datetimes(datetimes=datetimes))

    if (solar_config := config.solar_position) is not None:
        datetimes = _build_feature_datetimes(t0, solar_config)

        sample.update(
            _build_solar_features(datetimes, lon=location.longitude, lat=location.latitude)
        )

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


def add_source_arrays(
    sample: NumpySample,
    datasets_dict: SourceDict,
) -> None:
    """Add source data arrays to the sample in place.

    Generation timestamps are float64 POSIX seconds. Added arrays may share
    memory with the source arrays.

    Args:
        sample: Sample to populate.
        datasets_dict: Preprocessed source arrays for one sample.
    """
    for key in ("generation_input", "generation_target"):
        if key not in datasets_dict:
            continue

        da_gen = datasets_dict[key]

        # Generation is already normalised by capacity; capacity remains in MW.
        gen_params = list(da_gen["gen_param"].values)
        generation = da_gen.isel(gen_param=gen_params.index("generation_mw")).values
        capacity = da_gen.isel(gen_param=gen_params.index("capacity_mwp")).values

        sample.update(
            {
                key: generation,
                f"{key}_capacity_mwp": capacity,
                f"{key}_time_utc": get_posix_timestamp(da_gen["time_utc"].values),
            },
        )

    if "sat" in datasets_dict:
        sample["satellite"] = datasets_dict["sat"].values

    if "nwp" in datasets_dict:
        sample["nwp"] = {
            provider: da_nwp.values for provider, da_nwp in datasets_dict["nwp"].items()
        }


def add_source_metadata(
    metadata: dict[str, Any],
    datasets_dict: SourceDict,
) -> None:
    """Add source diagnostics to an existing metadata dictionary in place.

    Absolute timestamps are float64 POSIX seconds; NWP lead times are float32 hours.
    Coordinate arrays retain source units and dtypes and may share source memory.

    Args:
        metadata: Metadata dictionary to populate, retaining unrelated entries.
        datasets_dict: Source arrays for one sample.
    """
    if "sat" in datasets_dict:
        da_sat = datasets_dict["sat"]
        metadata.update({
            "satellite_time_utc": get_posix_timestamp(da_sat["time_utc"].values),
            "satellite_x_geostationary": da_sat["x_geostationary"].values,
            "satellite_y_geostationary": da_sat["y_geostationary"].values,
        })

    if "nwp" in datasets_dict:
        metadata["nwp"] = {}
        for provider, da_nwp in datasets_dict["nwp"].items():
            step_hours = (da_nwp["step"].values / np.timedelta64(1, "h")).astype(np.float32)
            metadata["nwp"][provider] = {
                "init_time_utc": get_posix_timestamp(da_nwp["init_time_utc"].values),
                "step_hours": step_hours,
            }


def _build_solar_features(
    datetimes: NDArray[np.datetime64],
    lon: float,
    lat: float,
) -> dict[str, NDArray[np.float32]]:
    """Creates NumpySample with standardized solar coordinates.

    Args:
        datetimes: Datetimes for which to calculate the solar coordinates.
        lon: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        lat: Latitude in decimal degrees. Positive north of equator, negative to south.
    """
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Scale azimuth from [0, 360] and elevation from [-90, 90] degrees to [0, 1].
    azimuth = azimuth / 360
    elevation = elevation / 180 + 0.5

    return {
        "solar_azimuth": azimuth.astype(np.float32),
        "solar_elevation": elevation.astype(np.float32),
    }
