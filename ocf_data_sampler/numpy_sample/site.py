"""Convert site to Numpy Sample"""

import xarray as xr


class SiteSampleKey:

    generation = "site"
    capacity_kwp = "site_capacity_kwp"
    time_utc = "site_time_utc"
    t0_idx = "site_t0_idx"
    solar_azimuth = "site_solar_azimuth"
    solar_elevation = "site_solar_elevation"
    id = "site_id"


def convert_site_to_numpy_sample(da: xr.DataArray, t0_idx: int | None = None) -> dict:
    """Convert from Xarray to NumpySample"""

    # Extract values from the DataArray
    input_variables = {
        'generation': da.values,
        'capacity_kwp': da.isel(time_utc=0)["capacity_kwp"].values,
        'time_utc': da["time_utc"].values.astype(float),
    }

    # Check for None values and raise error with specific variable name if value is not passed or None
    for var, value in input_variables.items():
        if value is None:
            raise ValueError(f"The variable '{var}' has a None value.")
    
    # Create the example dictionary
    example = {
        SiteSampleKey.generation: input_variables['generation'],
        SiteSampleKey.capacity_kwp: input_variables['capacity_kwp'],
        SiteSampleKey.time_utc: input_variables['time_utc'],
    }

    if t0_idx is not None:
        example[SiteSampleKey.t0_idx] = t0_idx

    return example
