import numpy as np
import pytest

from ocf_data_sampler.config import Configuration, load_yaml_configuration
from ocf_data_sampler.numpy_sample import (
    GSPSampleKey,
    NWPSampleKey,
    SatelliteSampleKey,
    SiteSampleKey,
)


def _create_base_sample(target_shape=(7,), target_key=None, target_value=None):
    """Create base numpy sample with common structure"""
    nwp_data = {
        "nwp": np.random.rand(4, 1, 2, 2),
        "x": np.array([1, 2]),
        "y": np.array([1, 2]),
        NWPSampleKey.channel_names: ["t"],
    }

    sat_shape = (7, 1, 2, 2)
    sample = {
        "nwp": {"ukv": nwp_data},
        SatelliteSampleKey.satellite_actual: np.random.rand(*sat_shape),
        "solar_azimuth": np.random.rand(*target_shape),
        "solar_elevation": np.random.rand(*target_shape),
    }

    if target_key and target_value is not None:
        sample[target_key] = target_value

    return sample


@pytest.fixture(scope="module")
def numpy_sample_site():
    """Synthetic site sample data"""
    shape = (7,)
    sample = _create_base_sample(shape, SiteSampleKey.generation, np.random.rand(*shape))
    sample.update({
        "date_cos": np.random.rand(*shape),
        "date_sin": np.random.rand(*shape),
        "time_cos": np.random.rand(*shape),
        "time_sin": np.random.rand(*shape),
    })
    return sample


@pytest.fixture
def numpy_sample_gsp():
    """Synthetic GSP sample data"""
    shape = (7,)
    return _create_base_sample(shape, GSPSampleKey.gsp, np.random.rand(*shape))


@pytest.fixture
def pvnet_configuration_object(pvnet_config_filename) -> Configuration:
    """Load configuration from temporary file path"""
    return load_yaml_configuration(pvnet_config_filename)
