import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import tempfile

from ocf_data_sampler.torch_datasets.datasets.pvnet_uk_regional import PVNetUKRegionalDataset
from ocf_data_sampler.config.save import save_yaml_configuration
from ocf_data_sampler.config.load import load_yaml_configuration
from ocf_data_sampler.numpy_sample import NWPSampleKey, GSPSampleKey, SatelliteSampleKey
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk_regional import process_and_combine_datasets, compute
from ocf_data_sampler.select.location import Location

def test_process_and_combine_datasets(pvnet_config_filename):

    # Load in config for function and define location 
    config = load_yaml_configuration(pvnet_config_filename)
    t0 = pd.Timestamp("2024-01-01 00:00")
    location = Location(coordinate_system="osgb", x=1234, y=5678, id=1)

    nwp_data = xr.DataArray(
        np.random.rand(4, 2, 2, 2),
        dims=["time_utc", "channel", "y", "x"],
        coords={
            "time_utc": pd.date_range("2024-01-01 00:00", periods=4, freq="h"),
            "channel": ["t2m", "dswrf"],
            "step": ("time_utc", pd.timedelta_range(start='0h', periods=4, freq='h')),
            "init_time_utc": pd.Timestamp("2024-01-01 00:00")
        }
    )

    sat_data = xr.DataArray(
        np.random.rand(7, 1, 2, 2),
        dims=["time_utc", "channel", "y", "x"],
        coords={
            "time_utc": pd.date_range("2024-01-01 00:00", periods=7, freq="5min"),
            "channel": ["HRV"],
            "x_geostationary": (["y", "x"], np.array([[1, 2], [1, 2]])),
            "y_geostationary": (["y", "x"], np.array([[1, 1], [2, 2]]))
        }
    )

    # Combine as dict
    dataset_dict = {
        "nwp": {"ukv": nwp_data},
        "sat": sat_data
    }

    # Call relevant function
    result = process_and_combine_datasets(dataset_dict, config, t0, location)

    # Assert result is dict - check and validate
    assert isinstance(result, dict)
    assert NWPSampleKey.nwp in result
    assert result[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
    assert result[NWPSampleKey.nwp]["ukv"][NWPSampleKey.nwp].shape == (4, 1, 2, 2)

def test_compute():
    """Test compute function with dask array"""
    da_dask = xr.DataArray(da.random.random((5, 5)))

    # Create a nested dictionary with dask array
    nested_dict = {
        "array1": da_dask,
        "nested": {
            "array2": da_dask
        }
    }

    # Ensure initial data is lazy - i.e. not yet computed
    assert not isinstance(nested_dict["array1"].data, np.ndarray)
    assert not isinstance(nested_dict["nested"]["array2"].data, np.ndarray)

    # Call the compute function
    result = compute(nested_dict)

    # Assert that the result is an xarray DataArray and no longer lazy
    assert isinstance(result["array1"], xr.DataArray)
    assert isinstance(result["nested"]["array2"], xr.DataArray)
    assert isinstance(result["array1"].data, np.ndarray)
    assert isinstance(result["nested"]["array2"].data, np.ndarray)

    # Ensure there no NaN values in computed data
    assert not np.isnan(result["array1"].data).any()
    assert not np.isnan(result["nested"]["array2"].data).any()

def test_pvnet(pvnet_config_filename):

    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317 # no of GSPs not including the National level
    # NB. I have not checked this value is in fact correct, but it does seem to stay constant
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317*39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    for key in [
        NWPSampleKey.nwp, SatelliteSampleKey.satellite_actual, GSPSampleKey.gsp,
        GSPSampleKey.solar_azimuth, GSPSampleKey.solar_elevation,
    ]:
        assert key in sample
    
    for nwp_source in ["ukv"]:
        assert nwp_source in sample[NWPSampleKey.nwp]

    # check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample[SatelliteSampleKey.satellite_actual].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample[NWPSampleKey.nwp]["ukv"][NWPSampleKey.nwp].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample[GSPSampleKey.gsp].shape == (7,)
    # Solar angles have same shape as GSP data
    assert sample[GSPSampleKey.solar_azimuth].shape == (7,)
    assert sample[GSPSampleKey.solar_elevation].shape == (7,)

def test_pvnet_no_gsp(pvnet_config_filename):

    # load config
    config = load_yaml_configuration(pvnet_config_filename)
    # remove gsp
    config.input_data.gsp.zarr_path = ''

    # save temp config file
    with tempfile.NamedTemporaryFile() as temp_config_file:
        save_yaml_configuration(config, temp_config_file.name)
        # Create dataset object
        dataset = PVNetUKRegionalDataset(temp_config_file.name)

        # Generate a sample
        _ = dataset[0]
