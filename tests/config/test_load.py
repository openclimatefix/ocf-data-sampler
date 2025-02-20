from ocf_data_sampler.config import Configuration, load_yaml_configuration


def test_load_yaml_configuration(test_config_filename):
    loaded_config = load_yaml_configuration(test_config_filename)
    assert isinstance(loaded_config, Configuration)
        
