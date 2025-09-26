import numpy as np
import pytest

from ocf_data_sampler.config import Configuration
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
    SiteSampleKey,
)


@pytest.fixture
def numpy_sample_site():
    """Synthetic data generation"""
    expected_site_shape = (7,)
    expected_nwp_ukv_shape = (4, 1, 2, 2)
    expected_sat_shape = (7, 1, 2, 2)
    expected_solar_shape = (7,)

    nwp_data = {
        "nwp": np.random.rand(*expected_nwp_ukv_shape),
        "x": np.array([1, 2]),
        "y": np.array([1, 2]),
        NWPSampleKey.channel_names: ["t"],
    }

    return {
        "nwp": {"ukv": nwp_data},
        SiteSampleKey.generation: np.random.rand(*expected_site_shape),
        SatelliteSampleKey.satellite_actual: np.random.rand(*expected_sat_shape),
        "solar_azimuth": np.random.rand(*expected_solar_shape),
        "solar_elevation": np.random.rand(*expected_solar_shape),
        "date_cos": np.random.rand(*expected_solar_shape),
        "date_sin": np.random.rand(*expected_solar_shape),
        "time_cos": np.random.rand(*expected_solar_shape),
        "time_sin": np.random.rand(*expected_solar_shape),
    }


@pytest.fixture
def numpy_sample_gsp():
    """Synthetic data generation"""
    expected_gsp_shape = (7,)
    expected_nwp_ukv_shape = (4, 1, 2, 2)
    expected_sat_shape = (7, 1, 2, 2)
    expected_solar_shape = (7,)

    nwp_data = {
        "nwp": np.random.rand(*expected_nwp_ukv_shape),
        "x": np.array([1, 2]),
        "y": np.array([1, 2]),
        NWPSampleKey.channel_names: ["t"],
    }

    return {
        "nwp": {
            "ukv": nwp_data,
        },
        GSPSampleKey.gsp: np.random.rand(*expected_gsp_shape),
        SatelliteSampleKey.satellite_actual: np.random.rand(*expected_sat_shape),
        "solar_azimuth": np.random.rand(*expected_solar_shape),
        "solar_elevation": np.random.rand(*expected_solar_shape),
    }


@pytest.fixture
def pvnet_configuration_object(pvnet_config_filename) -> Configuration:
    """Loads the configuration from the temporary file path."""
    return load_yaml_configuration(pvnet_config_filename)
