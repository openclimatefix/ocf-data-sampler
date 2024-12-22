import pytest
import numpy as np
import pandas as pd
import xarray as xr
from ocf_data_sampler.config import Configuration
from ocf_data_sampler.select.location import Location
from ocf_data_sampler.select.select_time_slice import select_time_slice_nwp

from ocf_data_sampler.torch_datasets.process_and_combine import (
    process_and_combine_datasets,
    merge_dicts,
    fill_nans_in_arrays,
    compute,
)
from ocf_data_sampler.numpy_batch import NWPBatchKey, SatelliteBatchKey, GSPBatchKey


@pytest.fixture
def mock_configuration():
    config = Configuration()
    config.input_data.nwp = {
        "ukv": type("Provider", (object,), {"provider": "ukv"}),
        "ecmwf": type("Provider", (object,), {"provider": "ecmwf"}),
    }
    config.input_data.gsp = type(
        "GSPConfig",
        (object,),
        {
            "interval_start_minutes": -180,
            "interval_end_minutes": 180,
            "time_resolution_minutes": 30,
        },
    )()
    config.input_data.site = type(
        "SiteConfig",
        (object,),
        {
            "interval_start_minutes": -120,
            "interval_end_minutes": 120,
            "time_resolution_minutes": 15,
        },
    )()
    return config


@pytest.fixture
def mock_dataset_dict():
    x_osgb = np.linspace(0, 10, 2)
    y_osgb = np.linspace(0, 10, 2)
    init_time_utc = pd.date_range("2023-01-01T00:00", periods=10, freq="1h")
    step = pd.to_timedelta(np.arange(10), unit="h")  # Create a valid step coordinate

    # Create mock NWP data with valid `step` and `init_time_utc`
    nwp_data = xr.DataArray(
        np.random.rand(10, 5, 2, 2),
        dims=["step", "channel", "x", "y"],
        coords={
            "step": step,  # Properly define `step` as a coordinate
            "init_time_utc": ("step", init_time_utc),  # Link `init_time_utc` to `step`
            "channel": ["cdcb", "lcc", "mcc", "hcc", "sde"],
            "x_osgb": ("x", x_osgb),
            "y_osgb": ("y", y_osgb),
        },
    )

    # Ensure step remains accessible even after dimension swapping
    nwp_data = nwp_data.swap_dims({"step": "init_time_utc"}).reset_coords("step", drop=False)

    # Create mock satellite data
    sat_data = xr.DataArray(
        np.random.rand(10, 1, 2, 2),
        dims=["time", "channel", "x", "y"],
        coords={
            "time": pd.date_range("2023-01-01", periods=10, freq="30min"),
            "channel": ["HRV"],
            "x_osgb": ("x", x_osgb),
            "y_osgb": ("y", y_osgb),
        },
    )

    # Create mock GSP data
    gsp_data = xr.DataArray(
        np.random.rand(10),
        dims=["time"],
        coords={"time": pd.date_range("2023-01-01", periods=10, freq="30min")},
    )
    gsp_future_data = xr.DataArray(
        np.random.rand(10),
        dims=["time"],
        coords={"time": pd.date_range("2023-01-01T05:00", periods=10, freq="30min")},
    )

    return {
        "nwp": {"ukv": nwp_data},
        "sat": sat_data,
        "gsp": gsp_data,
        "gsp_future": gsp_future_data,
    }


def test_process_and_combine_datasets(mock_configuration, mock_dataset_dict):
    location = Location(x=0, y=0, id=1)
    t0 = pd.Timestamp("2023-01-01 06:00")

    # Apply time slicing to the NWP data
    for nwp_key, da_nwp in mock_dataset_dict["nwp"].items():
        mock_dataset_dict["nwp"][nwp_key] = select_time_slice_nwp(
            da=da_nwp,
            t0=t0,
            interval_start=pd.Timedelta(hours=-3),
            interval_end=pd.Timedelta(hours=3),
            sample_period_duration=pd.Timedelta(minutes=30),
        )

    result = process_and_combine_datasets(
        dataset_dict=mock_dataset_dict,
        config=mock_configuration,
        t0=t0,
        location=location,
        target_key="gsp",
    )

    assert isinstance(result, dict)
    assert GSPBatchKey.gsp in result
    assert NWPBatchKey.nwp in result
    assert SatelliteBatchKey.satellite_actual in result


def test_merge_dicts():
    dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]
    merged = merge_dicts(dicts)
    assert merged == {"a": 1, "b": 3, "c": 4}


def test_fill_nans_in_arrays():
    batch = {
        "a": np.array([1.0, np.nan, 3.0]),
        "b": {"nested": np.array([np.nan, 5.0])},
    }
    filled_batch = fill_nans_in_arrays(batch)
    assert np.array_equal(filled_batch["a"], np.array([1.0, 0.0, 3.0]))
    assert np.array_equal(filled_batch["b"]["nested"], np.array([0.0, 5.0]))


def test_compute():
    data = xr.DataArray(
        np.random.rand(10), dims=["time"], coords={"time": pd.date_range("2023-01-01", periods=10)}
    )
    nested_dict = {"level1": {"level2": data}}
    computed_dict = compute(nested_dict)

    assert computed_dict["level1"]["level2"].equals(data)
