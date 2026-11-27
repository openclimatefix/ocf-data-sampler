import numpy as np
import pytest
from xarray import DataArray

from ocf_data_sampler.load.nwp import open_nwp


def test_open_nwp_osgb(nwp_ukv_zarr_path):
    da = open_nwp(zarr_path=nwp_ukv_zarr_path)
    assert isinstance(da, DataArray)
    # The underlying data should be lazy-loaded, not in memory
    assert not da.variable._in_memory
    assert da.dims == ("init_time_utc", "step", "channel", "x_osgb", "y_osgb")
    assert da.shape == (24 * 7, 11, 4, 50, 100)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_open_nwp_lonlat(nwp_ecmwf_zarr_path):
    da = open_nwp(zarr_path=nwp_ecmwf_zarr_path)
    assert isinstance(da, DataArray)
    # The underlying data should be lazy-loaded, not in memory
    assert not da.variable._in_memory
    assert da.dims == ("init_time_utc", "step", "channel", "longitude", "latitude")
    assert da.shape == (24 * 7, 15, 3, 15, 12)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_open_nwp_geostationary(nwp_cloudcasting_zarr_path):
    da = open_nwp(zarr_path=nwp_cloudcasting_zarr_path)
    assert isinstance(da, DataArray)
    # The underlying data should be lazy-loaded, not in memory
    assert not da.variable._in_memory
    assert da.dims == ("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary")
    assert "area" in da.attrs
    assert da.shape == (2, 12, 3, 100, 100)
    assert len(np.unique(da.coords["channel"])) == da.shape[2]


def test_open_nwp_rejects_unknown_coords(tmp_path):
    zarr_path = tmp_path / "ukv_x_y_coords.zarr"
    array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time_utc", "step", "channel", "x", "y"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "channel": ["t"],
            "x": [0],
            "y": [50],
        },
    )
    array.to_zarr(zarr_path)

    with pytest.raises(ValueError, match="Did not find any coordinate pairs"):
        open_nwp(zarr_path=str(zarr_path))


def test_open_nwp_integer_latitude(tmp_path):
    """Test integer spatial coordinates are accepted."""
    zarr_path = tmp_path / "integer_ecmwf_latitude.zarr"
    array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time_utc", "step", "variable", "longitude", "latitude"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "variable": ["t"],
            "longitude": np.array([0], dtype=np.float32),
            "latitude": [50],
        },
    )
    array.to_zarr(zarr_path)
    da = open_nwp(zarr_path=str(zarr_path))

    assert np.issubdtype(da.latitude.dtype, np.integer)


def test_open_nwp_rejects_uneven_steps(tmp_path):
    """Test steps must be evenly spaced - the valid-t0 calculation assumes it."""
    zarr_path = tmp_path / "uneven_ukv_steps.zarr"
    array = DataArray(
        np.random.rand(1, 3, 1, 1, 1).astype(np.float32),
        dims=("init_time_utc", "step", "channel", "x_osgb", "y_osgb"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            # Hourly, then 3-hourly
            "step": np.array([1, 2, 5], dtype="timedelta64[h]"),
            "channel": ["t"],
            "x_osgb": np.array([0], dtype=np.float32),
            "y_osgb": np.array([50], dtype=np.float32),
        },
    )
    array.to_zarr(zarr_path)

    with pytest.raises(ValueError, match="must be evenly spaced"):
        open_nwp(zarr_path=str(zarr_path))


def test_open_nwp_rejects_non_numeric_spatial_coord(tmp_path):
    """Test the detected spatial coordinates are validated as numeric."""
    zarr_path = tmp_path / "bad_ecmwf_longitude.zarr"
    bad_array = DataArray(
        np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
        dims=("init_time_utc", "step", "variable", "longitude", "latitude"),
        coords={
            "init_time_utc": [np.datetime64("2023-01-01")],
            "step": [np.timedelta64(1, "h")],
            "variable": ["t"],
            "longitude": ["west"],
            "latitude": np.array([50.0], dtype=np.float32),
        },
    )
    bad_array.to_zarr(zarr_path)
    with pytest.raises(
        TypeError,
        match="Coordinate 'longitude' in NWP data should be number",
    ):
        open_nwp(zarr_path=str(zarr_path))


def test_open_nwp_rejects_mismatched_geostationary_projections(tmp_path):
    """Test geostationary stores must share a projection before they are concatenated."""
    zarr_paths = []
    for i, lon_0 in enumerate([9.5, 0.0]):
        array = DataArray(
            np.random.rand(1, 1, 1, 1, 1).astype(np.float32),
            dims=("init_time_utc", "step", "channel", "x_geostationary", "y_geostationary"),
            coords={
                "init_time_utc": [np.datetime64("2023-01-01") + np.timedelta64(i, "h")],
                "step": [np.timedelta64(1, "h")],
                "channel": ["IR_097"],
                "x_geostationary": [0.0],
                "y_geostationary": [0.0],
            },
        )
        # The projection is dataset-level metadata, as in the cloudcasting stores
        ds = array.to_dataset(name="sat_pred")
        ds.attrs["area"] = {"satellite": {"projection": {"lon_0": lon_0}}}
        zarr_path = tmp_path / f"cloudcasting_{i}.zarr"
        ds.to_zarr(zarr_path)
        zarr_paths.append(str(zarr_path))

    with pytest.raises(ValueError, match="Multiple geostationary projections found"):
        open_nwp(zarr_path=zarr_paths)
