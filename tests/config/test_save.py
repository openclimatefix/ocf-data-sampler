"""Tests for configuration saving functionality."""
import pytest
from pathlib import Path
import tempfile
import yaml

from ocf_data_sampler.config import Configuration, save_yaml_configuration

@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

def test_save_yaml_configuration_basic(temp_dir):
    """Test basic configuration saving functionality."""
    config = Configuration()
    filepath = temp_dir / "config.yaml"
    result = save_yaml_configuration(config, filepath)
    
    assert filepath.exists()
    with open(filepath) as f:
        loaded_yaml = yaml.safe_load(f)
    assert isinstance(loaded_yaml, dict)

def test_save_yaml_configuration_none_filename():
    """Test that None filename raises ValueError."""
    config = Configuration()
    with pytest.raises(ValueError, match="filename cannot be None"):
        save_yaml_configuration(config, None)

def test_save_yaml_configuration_invalid_directory(temp_dir):
    """Test handling of invalid directory paths."""
    config = Configuration()
    invalid_path = (temp_dir / "nonexistent" / "config.yaml").resolve()
    with pytest.raises(ValueError, match="Directory does not exist"):
        save_yaml_configuration(config, invalid_path)
