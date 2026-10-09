"""Functions to convert xarray datasets to numpy samples."""

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.common.time_utils import date_range, get_posix_timestamp, minutes
from ocf_data_sampler.config.model import PVNetDataConfig
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
    """Convert data to numpy arrays and add auxiliary features.

    Note: the data in `dataset_dict` is expected to already be preprocessed - see
    `preprocess_dataset_dict`.

    Args:
        dataset_dict: Dictionary of xarray datasets
        t0: init-time for sample
        location: location of the sample
        config: PVNetDataConfig object
        include_extra_metadata: Whether to add additional non-essential metadata to the sample
    """
    # Convert all xarray modalities to a single NumpySample
    sample = convert_to_numpy_sample(dataset_dict, include_extra_metadata)

    sample["location_id"] = location.id

    if include_extra_metadata:
        sample["location_longitude"] = location.longitude
        sample["location_latitude"] = location.latitude

    # Add t0 embedding if configured
    if config.t0_embedding is not None:
        sample.update(
            make_t0_encoding_numpy_sample(t0, config.t0_embedding.embeddings),
        )

    # Add datetime encodings if configured
    if config.datetime_encoding is not None:
        dt_config = config.datetime_encoding

        datetimes = date_range(
            t0 + minutes(dt_config.interval_start_minutes),
            t0 + minutes(dt_config.interval_end_minutes),
            freq=minutes(dt_config.time_resolution_minutes),
        )
        sample.update(encode_datetimes(datetimes=datetimes))

    # Add solar position if configured
    if config.solar_position is not None:
        solar_config = config.solar_position

        # Create datetime range for solar position calculation
        datetimes = date_range(
            t0 + minutes(solar_config.interval_start_minutes),
            t0 + minutes(solar_config.interval_end_minutes),
            freq=minutes(solar_config.time_resolution_minutes),
        )

        sample.update(
            make_sun_position_numpy_sample(datetimes, lon=location.longitude, lat=location.latitude)
        )

    sample["t0"] = get_posix_timestamp(t0)

    return sample


def convert_to_numpy_sample(
    datasets_dict: SourceDict,
    include_extra_metadata: bool = False,
) -> NumpySample:
    """Convert a dictionary of xarray objects to a NumpySample.

    Args:
        datasets_dict: Dictionary of xarray DataArrays, with same structure as used inside
            PVNetDataset classes. Expected keys are any of following:
            - "generation_input": DataArray of generation data used as model input
            - "generation_target": DataArray of generation data used as the prediction target
            - "sat": DataArray of satellite data
            - "nwp": dict of DataArrays by provider name (e.g. {"ukv": da, "ecmwf": da})
        include_extra_metadata: Whether to add additional non-essential metadata to the batch

    Returns:
        NumpySample dictionary with all modalities merged
    """
    numpy_sample: NumpySample = {}

    for key in ("generation_input", "generation_target"):
        if key not in datasets_dict:
            continue

        da = datasets_dict[key]

        # generation_mw is expected to already been normalised by capacity so should be in the
        # range [0, 1]. capacity_mwp is still expected to be in MW
        gen_idx = list(da["gen_param"].values).index("generation_mw")
        cap_idx = list(da["gen_param"].values).index("capacity_mwp")

        numpy_sample.update(
            {
                key: da.isel(gen_param=gen_idx).values,
                f"{key}_capacity_mwp": da.isel(gen_param=cap_idx).values,
                f"{key}_time_utc": da["time_utc"].values.astype(float),
            },
        )

    if "sat" in datasets_dict:
        da = datasets_dict["sat"]
        numpy_sample.update({"satellite": da.values})

        if include_extra_metadata:
            numpy_sample.update(
                {
                    "satellite_time_utc": da["time_utc"].values.astype(float),
                    "satellite_x_geostationary": da["x_geostationary"].values,
                    "satellite_y_geostationary": da["y_geostationary"].values,
                },
            )

    if "nwp" in datasets_dict:
        for provider, da in datasets_dict["nwp"].items():
            nwp_key = f"nwp_{provider}"
            numpy_sample.update({nwp_key: da.values})

            if include_extra_metadata:
                step_hours = (da["step"].values / np.timedelta64(1, "h")).astype(float)
                target_times = (da["init_time_utc"].values + da["step"].values).astype(float)

                numpy_sample.update({
                    f"{nwp_key}_init_time_utc": da["init_time_utc"].values.astype(float),
                    f"{nwp_key}_step_hours": step_hours,
                    f"{nwp_key}_target_time_utc": target_times,
                })

    return numpy_sample


def make_sun_position_numpy_sample(
    datetimes: NDArray[np.datetime64],
    lon: float,
    lat: float,
) -> NumpySample:
    """Creates NumpySample with standardized solar coordinates.

    Args:
        datetimes: Datetimes for which to calculate the solar coordinates.
        lon: Longitude in decimal degrees. Positive east of prime meridian, negative to west.
        lat: Latitude in decimal degrees. Positive north of equator, negative to south.
    """
    azimuth, elevation = calculate_azimuth_and_elevation(datetimes, lon, lat)

    # Normalise
    # Azimuth is in range [0, 360] degrees
    azimuth = azimuth / 360

    # Elevation is in range [-90, 90] degrees
    elevation = elevation / 180 + 0.5

    return {
        "solar_azimuth": azimuth.astype(np.float32),
        "solar_elevation": elevation.astype(np.float32),
    }


def make_t0_encoding_numpy_sample(
    t0: np.datetime64,
    embeddings: list[tuple[str, str]],
) -> NumpySample:
    """Creates NumpySample with t0 time embeddings.

    Args:
        t0: The time to create sin-cos embeddings for
        embeddings: The periods to encode (e.g., "1h", "Nh", "1y", "Ny") and their representation
            (either "cyclic" or "linear"). When cyclic, the period is sin-cos embedded, else it is
            0-1 scaled as fraction through the period. Note that using "cyclic" adds 2 elements to
            the output array to embed a period whilst "linear" adds only 1 element.

    Returns:
        NumpySample with t0 time embeddings.
    """
    return {"t0_embedding": encode_t0(t0, embeddings)}
