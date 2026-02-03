import numpy as np
import xarray as xr


def test_ukv_rename_logic_legacy():
    """Test legacy renaming."""
    ds = xr.Dataset(
        coords={
            "x": np.array([1, 2]),
            "y": np.array([3, 4]),
            "init_time": np.array([0]),
            "variable": np.array(["temp"]),
        },
    )

    rename_map = {
        "init_time": "init_time_utc",
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }

    # Broken into two lines to stay under 100 characters
    actual_rename = {
        k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords
    }
    ds_renamed = ds.rename(actual_rename)

    assert "x_osgb" in ds_renamed.coords
    assert "y_osgb" in ds_renamed.coords
    assert "x" not in ds_renamed.coords


def test_ukv_rename_logic_new():
    """Test new naming convention."""
    ds = xr.Dataset(
        coords={
            "x_osgb": np.array([1, 2]),
            "y_osgb": np.array([3, 4]),
            "init_time_utc": np.array([0]),
            "channel": np.array(["temp"]),
        },
    )

    rename_map = {
        "init_time": "init_time_utc",
        "variable": "channel",
        "x": "x_osgb",
        "y": "y_osgb",
    }

    actual_rename = {
        k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords
    }
    ds_renamed = ds.rename(actual_rename)

    assert "x_osgb" in ds_renamed.coords
    assert "x" not in ds_renamed.coords
