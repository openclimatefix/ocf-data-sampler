from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.config.model import PVNetDataConfig


def test_load_yaml_configuration(config_filename):
    assert isinstance(load_yaml_configuration(config_filename), PVNetDataConfig)
