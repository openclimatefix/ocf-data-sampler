"""Tests - channel validation utility function"""

import pytest
import xarray as xr

from ocf_data_sampler.torch_datasets.utils.validate_channels import validate_channels


@pytest.fixture
def create_mock_xr_array():
    def _create(channels):
        return xr.DataArray([1.0 for _ in channels], coords={"channel": channels})
    return _create


class TestChannelValidation:
    @pytest.mark.parametrize("test_case", [
        # Success case - all channels match
        {
            "data_channels": ["t2m", "dswrf"],
            "norm_channels": ["t2m", "dswrf", "extra"],
            "source_name": "ecmwf",
            "expect_error": False
        },
        # Missing channel case
        {
            "data_channels": ["t2m", "missing_channel"],
            "norm_channels": ["t2m"],
            "source_name": "ecmwf",
            "expect_error": True,
            "error_match": "following channels for ecmwf are missing in normalisation means"  # Updated to match actual error
        },
        # Satellite case
        {
            "data_channels": ["IR_016", "VIS006"],
            "norm_channels": ["IR_016", "VIS006", "extra"],
            "source_name": "satellite",
            "expect_error": False
        },
        # Missing satellite channel case
        {
            "data_channels": ["IR_016", "missing_channel"],
            "norm_channels": ["IR_016"],
            "source_name": "satellite",
            "expect_error": True,
            "error_match": "following channels for satellite are missing in normalisation means"  # Updated to match actual error
        }
    ])
    def test_channel_validation(self, test_case, create_mock_xr_array):
        """Test channel validation for both NWP and satellite data"""
        data_channels = set(test_case["data_channels"])
        norm_array = create_mock_xr_array(test_case["norm_channels"])
        norm_channels = set(norm_array.channel.values)

        if test_case["expect_error"]:
            with pytest.raises(ValueError, match=test_case["error_match"]):
                validate_channels(
                    data_channels=data_channels,
                    means_channels=norm_channels,
                    stds_channels=norm_channels,
                    source_name=test_case["source_name"]
                )
        else:
            # Should not raise any exceptions
            validate_channels(
                data_channels=data_channels,
                means_channels=norm_channels,
                stds_channels=norm_channels,
                source_name=test_case["source_name"]
            )
