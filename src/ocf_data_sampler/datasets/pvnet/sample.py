"""Functions to convert xarray datasets to numpy samples."""

import numpy as np
from numpy.typing import NDArray

from ocf_data_sampler.features.solar import calculate_azimuth_and_elevation
from ocf_data_sampler.features.time_encodings import encode_t0
from ocf_data_sampler.datasets.pvnet.types import NumpySample, SourceDict


def convert_to_numpy_sample(
    datasets_dict: SourceDict,
    t0_idx: int,
    include_extra_metadata: bool = False,
) -> NumpySample:
    """Convert a dictionary of xarray objects to a NumpySample.

    Args:
        datasets_dict: Dictionary of xarray DataArrays, with same structure as used inside
            PVNetDataset classes. Expected keys are any of following:
            - "generation": DataArray of generation data
            - "sat": DataArray of satellite data
            - "nwp": dict of DataArrays by provider name (e.g. {"ukv": da, "ecmwf": da})
        t0_idx: Index of t0 within generation
        include_extra_metadata: Whether to add additional non-essential metadata to the batch

    Returns:
        NumpySample dictionary with all modalities merged
    """
    numpy_sample: NumpySample = {}

    if "generation" in datasets_dict:
        da = datasets_dict["generation"]

        # Get the position index of the generation and capacities
        gen_idx = np.argmax(da["gen_param"].values == "generation_mw")
        cap_idx = np.argmax(da["gen_param"].values == "capacity_mwp")

        generation_values = da.isel(gen_param=gen_idx).values
        capacity_value = da.isel(gen_param=cap_idx).values[0]

        if capacity_value!=0:
            generation_values = generation_values/capacity_value

        numpy_sample.update(
            {
                "generation": generation_values,
                "capacity_mwp": capacity_value,
                "generation_t0_idx": int(t0_idx),
                "generation_time_utc": da["time_utc"].values.astype(float),
            },
        )

        if include_extra_metadata:
            numpy_sample.update(
                {
                    "location_longitude": float(da["longitude"].values),
                    "location_latitude": float(da["latitude"].values),
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
        "solar_azimuth": azimuth,
        "solar_elevation": elevation,
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
