from ocf_data_sampler.config import Configuration, load_yaml_configuration


def test_load_yaml_configuration(config_filename):
    assert isinstance(load_yaml_configuration(config_filename), Configuration)
