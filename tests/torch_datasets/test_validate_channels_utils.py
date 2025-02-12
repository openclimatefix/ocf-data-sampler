"""Tests for channel validation utility functions"""

import pytest
import xarray as xr
from dataclasses import dataclass
from typing import List, Dict

from ocf_data_sampler.torch_datasets.utils.validate_channels import (
    validate_nwp_channels,
    validate_sat_channels
)


# Mock testing objects defined
@dataclass
class NWPConfig:
    provider: str
    channels: List[str]

@dataclass
class SatelliteConfig:
    channels: List[str]


@dataclass
class InputDataConfig:
    nwp: Dict[str, NWPConfig]
    satellite: SatelliteConfig = None


@dataclass
class Configuration:
    input_data: InputDataConfig


@pytest.fixture
def create_mock_xr_array():
    def _create(channels):
        return xr.DataArray([1.0 for _ in channels], coords={"channel": channels})
    return _create


class TestChannelValidation:
    @pytest.mark.parametrize("test_case", [
        {
            "config": {"ecmwf": ["t2m", "dswrf", "tcc"], "ukv": ["t", "dswrf"]},
            "constants": {
                "ecmwf": ["t2m", "dswrf", "tcc", "extra"], 
                "ukv": ["t", "dswrf", "extra"]
            },
            "expect_error": False
        },
        # Missing provider case
        {
            "config": {"ecmwf": ["t2m"]},
            "constants": {"ukv": ["t"]},
            "expect_error": True,
            "error_match": "Provider ecmwf not found"
        },
        # Missing channel case
        {
            "config": {"ecmwf": ["t2m", "missing_channel"]},
            "constants": {"ecmwf": ["t2m"]},
            "expect_error": True,
            "error_match": "following channels for ecmwf are missing in NWP_MEANS"
        }
    ])
    def test_nwp_validation(self, test_case, create_mock_xr_array):
        nwp_configs = {
            key: NWPConfig(provider=key, channels=channels)
            for key, channels in test_case["config"].items()
        }
        config = Configuration(input_data=InputDataConfig(nwp=nwp_configs))

        constants = {
            key: create_mock_xr_array(channels)
            for key, channels in test_case["constants"].items()
        }

        if test_case["expect_error"]:
            with pytest.raises(ValueError, match=test_case["error_match"]):
                validate_nwp_channels(config, constants, constants)
        else:
            validate_nwp_channels(config, constants, constants)

    @pytest.mark.parametrize("test_case", [
        {
            "channels": ["IR_016", "VIS006"],
            "constants": ["IR_016", "VIS006", "extra"],
            "expect_error": False
        },
        {
            "channels": ["IR_016", "missing_channel"],
            "constants": ["IR_016"],
            "expect_error": True,
            "error_match": "following satellite channels are missing in RSS_MEANS"
        },
        {
            "channels": None,
            "constants": ["IR_016"],
            "expect_error": False
        }
    ])
    def test_sat_validation(self, test_case, create_mock_xr_array):
        satellite_config = None
        if test_case["channels"] is not None:
            satellite_config = SatelliteConfig(channels=test_case["channels"])
        config = Configuration(input_data=InputDataConfig(nwp={}, satellite=satellite_config))

        constants = create_mock_xr_array(test_case["constants"])

        if test_case["expect_error"]:
            with pytest.raises(ValueError, match=test_case["error_match"]):
                validate_sat_channels(config, constants, constants)
        else:
            validate_sat_channels(config, constants, constants)
