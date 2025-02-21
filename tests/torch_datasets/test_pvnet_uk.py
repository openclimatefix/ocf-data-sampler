import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from ocf_data_sampler.config import load_yaml_configuration, save_yaml_configuration
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import (
    PVNetUKConcurrentDataset,
    PVNetUKRegionalDataset,
    compute,
    process_and_combine_datasets,
)



def test_process_and_combine_datasets(pvnet_config_filename, ds_nwp_ukv_time_sliced):

    config = load_yaml_configuration(pvnet_config_filename)
    
    t0 = pd.Timestamp("2024-01-01 00:00")
    location = Location(coordinate_system="osgb", x=1234, y=5678, id=1)

    sat_data = xr.DataArray(
        np.random.rand(7, 1, 2, 2),
        dims=["time_utc", "channel", "y", "x"],
        coords={
            "time_utc": pd.date_range("2024-01-01 00:00", periods=7, freq="5min"),
            "channel": ["HRV"],
            "x_geostationary": (["y", "x"], np.array([[1, 2], [1, 2]])),
            "y_geostationary": (["y", "x"], np.array([[1, 1], [2, 2]])),
        },
    )

    dataset_dict = {
        "nwp": {"ukv": ds_nwp_ukv_time_sliced},
        "sat": sat_data
    }

    sample = process_and_combine_datasets(dataset_dict, config, t0, location)

    assert isinstance(sample, dict)
    assert "nwp" in sample
    assert sample["satellite_actual"].shape == sat_data.shape
    assert sample["nwp"]["ukv"]["nwp"].shape == ds_nwp_ukv_time_sliced.shape
    assert "gsp_id" in sample


def test_compute():
    """Test compute function with dask array"""
    da_dask = xr.DataArray(dask.array.random.random((5, 5)))

    # Create a nested dictionary with dask array
    lazy_data_dict = {
        "array1": da_dask,
        "nested": {
            "array2": da_dask,
        },
    }

    computed_data_dict = compute(lazy_data_dict)

    # Assert that the result is no longer lazy
    assert isinstance(computed_data_dict["array1"].data, np.ndarray)
    assert isinstance(computed_data_dict["nested"]["array2"].data, np.ndarray)


def test_pvnet_uk_regional_dataset(pvnet_config_filename):

    # Create dataset object
    dataset = PVNetUKRegionalDataset(pvnet_config_filename)

    assert len(dataset.locations) == 317 # Number of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 317*39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    for key in [
        "nwp", "satellite_actual", "gsp",
        "gsp_solar_azimuth", "gsp_solar_elevation",
    ]:
        assert key in sample

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite_actual"].shape == (7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["gsp"].shape == (7,)
    # Solar angles have same shape as GSP data
    assert sample["gsp_solar_azimuth"].shape == (7,)
    assert sample["gsp_solar_elevation"].shape == (7,)


def test_pvnet_no_gsp(tmp_path, pvnet_config_filename):

    # Create new config without GSP inputs
    config = load_yaml_configuration(pvnet_config_filename)
    config.input_data.gsp.zarr_path = ""
    new_config_path = tmp_path / "pvnet_config_no_gsp.yaml"
    save_yaml_configuration(config, new_config_path)

    # Create dataset object
    dataset = PVNetUKRegionalDataset(new_config_path)

    # Generate a sample
    _ = dataset[0]


def test_pvnet_uk_concurrent_dataset(pvnet_config_filename):

    # Create dataset object using a limited set of GSPs for test
    gsp_ids = [1,2,3]
    num_gsps = len(gsp_ids)

    dataset = PVNetUKConcurrentDataset(pvnet_config_filename, gsp_ids=gsp_ids)

    assert len(dataset.locations) == num_gsps # Number of regional GSPs
    # NB. I have not checked the value (39 below) is in fact correct
    assert len(dataset.valid_t0_times) == 39
    assert len(dataset) == 39

    # Generate a sample
    sample = dataset[0]

    assert isinstance(sample, dict)

    for key in [
        "nwp", "satellite_actual", "gsp",
        "gsp_solar_azimuth", "gsp_solar_elevation",
    ]:
        assert key in sample

    for nwp_source in ["ukv"]:
        assert nwp_source in sample["nwp"]

    # Check the shape of the data is correct
    # 30 minutes of 5 minute data (inclusive), one channel, 2x2 pixels
    assert sample["satellite_actual"].shape == (num_gsps, 7, 1, 2, 2)
    # 3 hours of 60 minute data (inclusive), one channel, 2x2 pixels
    assert sample["nwp"]["ukv"]["nwp"].shape == (num_gsps, 4, 1, 2, 2)
    # 3 hours of 30 minute data (inclusive)
    assert sample["gsp"].shape == (num_gsps, 7)
    # Solar angles have same shape as GSP data
    assert sample["gsp_solar_azimuth"].shape == (num_gsps, 7)
    assert sample["gsp_solar_elevation"].shape == (num_gsps, 7)
