"""Tests for channel validation utility functions"""

import pytest
from ocf_data_sampler.torch_datasets.utils.validate_channels import (
    validate_channels,
    validate_nwp_channels,
    validate_satellite_channels,
)


class TestChannelValidation:
    """Tests for channel validation functions"""
    
    @pytest.mark.parametrize("test_case", [
        # Base validation - success case
        {
            "data_channels": ["channel1", "channel2"],
            "norm_channels": ["channel1", "channel2", "extra"],
            "source_name": "test_source",
            "expect_error": False
        },
        # Base validation - error case
        {
            "data_channels": ["channel1", "missing_channel"],
            "norm_channels": ["channel1"],
            "source_name": "test_source",
            "expect_error": True,
            "error_match": "following channels for test_source are missing in normalisation means"
        },
        # NWP case - success
        {
            "data_channels": ["t2m", "dswrf"],
            "norm_channels": ["t2m", "dswrf", "extra"],
            "source_name": "ecmwf",
            "expect_error": False
        },
        # NWP case - error
        {
            "data_channels": ["t2m", "missing_channel"],
            "norm_channels": ["t2m"],
            "source_name": "ecmwf",
            "expect_error": True,
            "error_match": "following channels for ecmwf are missing in normalisation means"
        },
        # Satellite case - success
        {
            "data_channels": ["IR_016", "VIS006"],
            "norm_channels": ["IR_016", "VIS006", "extra"],
            "source_name": "satellite",
            "expect_error": False
        },
        # Satellite case - error
        {
            "data_channels": ["IR_016", "missing_channel"],
            "norm_channels": ["IR_016"],
            "source_name": "satellite",
            "expect_error": True,
            "error_match": "following channels for satellite are missing in normalisation means"
        }
    ])
    def test_channel_validation(self, test_case):
        """Test channel validation for both base, NWP and satellite data"""
        if test_case["expect_error"]:
            with pytest.raises(ValueError, match=test_case["error_match"]):
                validate_channels(
                    data_channels=test_case["data_channels"],
                    means_channels=test_case["norm_channels"],
                    stds_channels=test_case["norm_channels"],
                    source_name=test_case["source_name"]
                )
        else:
            # Should not raise any exceptions
            validate_channels(
                data_channels=test_case["data_channels"],
                means_channels=test_case["norm_channels"],
                stds_channels=test_case["norm_channels"],
                source_name=test_case["source_name"]
            )
